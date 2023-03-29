/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <algorithm>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

/*
Get real sorted dim
eg: getRealDims({-1,1,2,3,-2}, 5) -> {1,2,3,4}
*/
std::vector<int64_t> getRealDims(std::vector<int64_t> input_dim, int64_t t_dim) {
    // handle negative dims
    for (int64_t i = 0; i < input_dim.size(); ++i) {
        if (input_dim[i] < 0) {
            input_dim[i] = input_dim[i] + t_dim;
        }
    }
    // remove duplicate dims and sort them
    std::vector<int64_t> dim_vec(input_dim);
    std::set<int64_t> s(dim_vec.begin(), dim_vec.end());
    dim_vec.assign(s.begin(), s.end());
    return dim_vec;
}

std::vector<int> infer_desc_shape(std::vector<int64_t> input_dim, std::vector<int64_t> reduce_dim, bool keepdim) {
    std::vector<int> output_dim(input_dim.begin(), input_dim.end());
    if (input_dim.size() == 0) {
        return output_dim;
    }
    int num = 0;
    for (auto i : reduce_dim) {
        if (keepdim) {
            output_dim[i] = 1;
        } else {
            auto it = output_dim.begin() + i - num;
            output_dim.erase(it);
            num++;
        }
    }
    return output_dim;
}

std::map<cnnlReduceOp_t, std::vector<diopiDtype_t>> supported_type_table = {
    {CNNL_REDUCE_ADD, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_AVG, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_MUL, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int64, diopi_dtype_float64}},
    {CNNL_REDUCE_MAX, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_MIN, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_AND, {diopi_dtype_bool, diopi_dtype_uint8, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_OR, {diopi_dtype_bool, diopi_dtype_uint8, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_NORM1, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}},
    {CNNL_REDUCE_NORM2, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64}}};

inline diopiDtype_t find_supported_type(cnnlReduceOp_t mode, diopiDtype_t input_type) {
    auto table = supported_type_table.find(mode);
    auto type = find(table->second.begin(), table->second.end(), input_type);
    if (type == table->second.end()) {
        // TODO(waiting dispatch): log system to log once only
        // std::cout << "ReduceOp input dtype is not supported, will cast to float for caculation." << std::endl;
        return diopi_dtype_float32;
    }
    return *type;
}

diopiError_t reduce_internal(diopiContextHandle_t ctx,
                             DiopiTensor& input_tr,
                             DiopiTensor& output_tr,
                             DiopiTensor& index_tr,
                             const std::vector<int64_t> reduce_dim,
                             cnnlReduceOp_t reduce_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DIOPI_CHECK(input_tr.numel() > 0, "operation does not have an identity.");
    DIOPI_CHECK(input_tr.is_contiguous(), "input tensor should be contiguous");

    CnnlReduceDescriptor reduce_desc;
    CnnlTensorDesc input_desc;
    CnnlTensorDesc output_desc;
    CnnlTensorDesc index_desc;

    cnnlDataType_t cnnl_dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&cnnl_dtype, input_tr.dtype()));
    if (reduce_op == CNNL_REDUCE_AVG && input_tr.dtype() == diopi_dtype_float16) {
        cnnl_dtype = CNNL_DTYPE_FLOAT;
    }

    // Only Min and Max Ops have indices as result.when reduce_dim > 1,
    auto reduce_indices =
        ((reduce_op == CNNL_REDUCE_MAX || reduce_op == CNNL_REDUCE_MIN) && reduce_dim.size() == 1) ? CNNL_REDUCE_FLATTENED_INDICES : CNNL_REDUCE_NO_INDICES;

    if (reduce_dim.size() == 0 || reduce_dim.size() == input_tr.dim()) {
        /* FULL-REDUCE: axis = [-1] instead of [0, 1, 2, ..., n] */
        std::vector<int64_t> full_reduce(1, -1);
        std::vector<int64_t> fake_size(input_tr.dim(), 1);
        reduce_desc.set(input_tr, full_reduce, reduce_op, reduce_indices, CNNL_32BIT_INDICES, cnnl_dtype);
        input_desc.set(input_tr, CNNL_LAYOUT_ARRAY);
        DiopiTensor fake_tensor = requiresTensor(ctx, fake_size, output_tr.dtype());
        output_desc.set(fake_tensor, CNNL_LAYOUT_ARRAY);
        DiopiTensor fake_tensor2 = requiresTensor(ctx, fake_size, index_tr.dtype());
        // index_desc.set_reduce(fake_tensor2);
        index_desc.set(fake_tensor2, CNNL_LAYOUT_ARRAY);
    } else {
        reduce_desc.set(input_tr, reduce_dim, reduce_op, reduce_indices, CNNL_32BIT_INDICES, cnnl_dtype);
        input_desc.set(input_tr, CNNL_LAYOUT_ARRAY);
        auto desc_shape = infer_desc_shape(input_tr.shape(), reduce_dim, true);
        output_desc.set(output_tr, CNNL_LAYOUT_ARRAY, desc_shape);
        index_desc.set(index_tr, CNNL_LAYOUT_ARRAY, desc_shape);
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, input_desc.get(), output_desc.get(), reduce_desc.get(), &workspace_size));
    void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

    DIOPI_CALLCNNL(cnnlReduce(handle,
                              reduce_desc.get(),
                              workspace_ptr,
                              workspace_size,
                              nullptr,
                              input_desc.get(),
                              input_tr.data(),
                              sizeof(int) * index_tr.numel(),
                              reduce_indices != CNNL_REDUCE_NO_INDICES ? index_tr.data() : nullptr,
                              nullptr,
                              output_desc.get(),
                              output_tr.data()));

    return diopiSuccess;
}

diopiError_t reduce_impl(diopiContextHandle_t ctx, DiopiTensor& output_tr, DiopiTensor& index_tr, DiopiTensor& input_tr, cnnlReduceOp_t reduce_op) {
    std::vector<int64_t> reduce_dim;
    for (int64_t i = 0; i <= input_tr.dim(); i++) {
        reduce_dim.push_back(i);
    }
    auto reduce_dtype = find_supported_type(reduce_op, input_tr.dtype());
    if (input_tr.dtype() != reduce_dtype) {
        input_tr = dataTypeCast(ctx, input_tr, reduce_dtype);
    }
    if (output_tr.dtype() != input_tr.dtype()) {
        auto output_tr_t = requiresTensor(ctx, output_tr.shape(), input_tr.dtype());
        DIOPI_CALL(reduce_internal(ctx, input_tr, output_tr_t, index_tr, reduce_dim, reduce_op));
        dataTypeCast(ctx, output_tr, output_tr_t);
    } else {
        DIOPI_CALL(reduce_internal(ctx, input_tr, output_tr, index_tr, reduce_dim, reduce_op));
    }
    return diopiSuccess;
}

diopiError_t reduce_dim_impl(diopiContextHandle_t ctx,
                             DiopiTensor& output_tr,
                             DiopiTensor& index_tr,
                             DiopiTensor& input_tr,
                             const std::vector<int64_t> dim_vec,
                             const bool keepdim,
                             cnnlReduceOp_t reduce_op) {
    std::vector<int64_t> reduce_dim = getRealDims(dim_vec, input_tr.dim());
    auto reduce_dtype = find_supported_type(reduce_op, input_tr.dtype());

    if (input_tr.dtype() != reduce_dtype) {
        input_tr = dataTypeCast(ctx, input_tr, reduce_dtype);
    }
    if (output_tr.dtype() != input_tr.dtype()) {
        auto output_tr_t = requiresTensor(ctx, output_tr.shape(), input_tr.dtype());
        DIOPI_CALL(reduce_internal(ctx, input_tr, output_tr_t, index_tr, reduce_dim, reduce_op));
        dataTypeCast(ctx, output_tr, output_tr_t);
    } else {
        DIOPI_CALL(reduce_internal(ctx, input_tr, output_tr, index_tr, reduce_dim, reduce_op));
    }
    return diopiSuccess;
}

extern "C" {

diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(out);
    auto index_tr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    diopiDtype_t dtype = output_tr.dtype();

    std::vector<int64_t> dim_vec(dim.data, dim.data + dim.len);

    DIOPI_CHECK(dtype != diopi_dtype_int64, "Sum: dtype == int64 is not supported in camb now");

    if (input_tr.dtype() != dtype) {
        input_tr = dataTypeCast(ctx, input_tr, dtype);
    }

    reduce_dim_impl(ctx, output_tr, index_tr, input_tr, dim_vec, false, CNNL_REDUCE_ADD);

    return diopiSuccess;
}

diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(out);
    auto index_tr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    diopiDtype_t dtype = output_tr.dtype();

    std::vector<int64_t> dim_vec(dim.data, dim.data + dim.len);

    if (input_tr.dtype() != dtype) {
        input_tr = dataTypeCast(ctx, input_tr, dtype);
    }

    reduce_dim_impl(ctx, output_tr, index_tr, input_tr, dim_vec, false, CNNL_REDUCE_AVG);
    return diopiSuccess;
}

diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(out);
    auto index_tr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    diopiDtype_t type = output_tr.dtype();

    if (input_tr.dtype() != type) {
        input_tr = dataTypeCast(ctx, input_tr, type);
    }

    reduce_dim_impl(ctx, output_tr, index_tr, input_tr, {*dim}, false, CNNL_REDUCE_MUL);
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input, int64_t dim) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(min);
    auto index_tr = DiopiTensor(min_indices);
    // Note: camb index out is int32 dtype
    auto index_tr_t = requiresTensor(ctx, index_tr.shape(), diopi_dtype_int32);

    reduce_dim_impl(ctx, output_tr, index_tr_t, input_tr, {dim}, false, CNNL_REDUCE_MIN);

    dataTypeCast(ctx, index_tr, index_tr_t);
    return diopiSuccess;
}

diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(min);
    auto index_tr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    reduce_impl(ctx, output_tr, index_tr, input_tr, CNNL_REDUCE_MIN);

    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(max);
    auto index_tr = DiopiTensor(max_indices);
    auto index_tr_t = requiresTensor(ctx, index_tr.shape(), diopi_dtype_int32);

    reduce_dim_impl(ctx, output_tr, index_tr_t, input_tr, {dim}, false, CNNL_REDUCE_MAX);

    dataTypeCast(ctx, index_tr, index_tr_t);
    return diopiSuccess;
}

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(max);
    auto index_tr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    reduce_impl(ctx, output_tr, index_tr, input_tr, CNNL_REDUCE_MAX);

    return diopiSuccess;
}

diopiError_t diopiNorm(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    float norm = p->fval;
    if (DiopiDataType().isInteger(p->stype)) norm = p->ival;
    DIOPI_CHECK(norm == 1.0 || norm == 2.0, "camb only support L1-Norm as p=1.0 and L2-Norm as p=2.0");

    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(out);
    auto index_tr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    diopiDtype_t dtype = output_tr.dtype();

    if (input_tr.dtype() != dtype) {
        input_tr = dataTypeCast(ctx, input_tr, dtype);
    }

    std::vector<int64_t> dim_vec(dim.data, dim.data + dim.len);
    reduce_dim_impl(ctx, output_tr, index_tr, input_tr, dim_vec, false, norm == 1.0 ? CNNL_REDUCE_NORM1 : CNNL_REDUCE_NORM2);

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
