#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
static std::vector<int> getPerm(DiopiTensor tensor, int64_t dim0, int64_t dim1) {
    int input_size = tensor.shape().size();
    if (dim0 < 0) {
        dim0 = dim0 + input_size;
    }
    if (dim1 < 0) {
        dim1 = dim1 + input_size;
    }

    std::vector<int> perms(input_size);
    std::iota(perms.begin(), perms.end(), 0);
    perms[dim0] = dim1;
    perms[dim1] = dim0;
    return perms;
}

static std::vector<int64_t> inferSize(std::vector<int64_t> batch_tensor1, std::vector<int64_t> batch_tensor2) {
    if (batch_tensor1.size() < batch_tensor2.size()) {
        batch_tensor1.insert(batch_tensor1.begin(), batch_tensor2.size() - batch_tensor1.size(), 1);
    } else if (batch_tensor1.size() > batch_tensor2.size()) {
        batch_tensor2.insert(batch_tensor2.begin(), batch_tensor1.size() - batch_tensor2.size(), 1);
    }

    std::vector<int64_t> res(batch_tensor1);
    for (int i = 0; i < batch_tensor1.size(); i++) {
        if (1 == batch_tensor1[i]) {
            res[i] = batch_tensor2[i];
        }
    }

    return res;
}

static int64_t multiplyIntegers(std::vector<int64_t> tensor) {
    int64_t out = 1;
    for (int i = 0; i < tensor.size(); i++) {
        out = out * tensor[i];
    }

    return out;
}

static diopiError_t vectorMulVector(diopiContextHandle_t ctx, DiopiTensor out_tensor, DiopiTensor vector1_tensor, DiopiTensor vector2_tensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    if (vector1_tensor.dtype() != diopi_dtype_float32 && vector1_tensor.dtype() != diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, vector1_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, vector2_tensor, diopi_dtype_float32));
    }

    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vector1Desc(vector1_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vector2Desc(vector2_tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor temp_out = requiresTensor(ctx, vector1_tensor.shape(), vector1_tensor.dtype());
    CnnlTensorDesc temp_outDesc(temp_out, CNNL_LAYOUT_ARRAY);

    std::vector<cnnlTensorDescriptor_t> inputs_desc(2);
    inputs_desc[0] = vector1Desc.get();
    inputs_desc[1] = vector2Desc.get();
    std::vector<const void*> inputs(2);
    inputs[0] = vector1_tensor.data();
    inputs[1] = vector2_tensor.data();

    DIOPI_CALLCNNL(cnnlMulN(handle, inputs_desc.data(), inputs.data(), 2, temp_outDesc.get(), temp_out.data()));
    int64_t dim_data = 0;
    diopiSize_t dim = {&dim_data, 1};

    if (out_tensor.dtype() == vector1_tensor.dtype()) {
        DIOPI_CALL(diopiSum(ctx, (diopiTensorHandle_t)out_tensor, (diopiTensorHandle_t)temp_out, dim));
    } else {
        DiopiTensor out32_tensor = requiresTensor(ctx, out_tensor.shape(), vector1_tensor.dtype());
        DIOPI_CALL(diopiSum(ctx, (diopiTensorHandle_t)out32_tensor, (diopiTensorHandle_t)temp_out, dim));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out32_tensor));
    }
    return diopiSuccess;
}

static diopiError_t matMulMat(diopiContextHandle_t ctx, DiopiTensor out, DiopiTensor input, DiopiTensor other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    if (input.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, other, diopi_dtype_float32));
    }

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(other, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmulDesc;
    cnnlMatMulDescriptor_t matmul_desc = matmulDesc.get();
    int32_t allow_tf32_i32 = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32, &(allow_tf32_i32), sizeof(int32_t)));
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> matmulAlgo;
    cnnlMatMulAlgo_t algo = matmulAlgo.get();

    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> matMulHeuristic;
    cnnlMatMulHeuristicResult_t heuristicResult = matMulHeuristic.get();
    int returnAlgoCount = 0;
    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(
        handle, matmul_desc, inputDesc.get(), otherDesc.get(), outDesc.get(), outDesc.get(), nullptr, 1, &heuristicResult, &returnAlgoCount));
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristicResult, algo, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    float alpha = 1;
    float beta = 0;
    if (out.dtype() == input.dtype()) {
        DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                     matmul_desc,
                                     algo,
                                     &alpha,
                                     inputDesc.get(),
                                     input.data(),
                                     otherDesc.get(),
                                     other.data(),
                                     &beta,
                                     outDesc.get(),
                                     out.data(),
                                     workspace,
                                     workspace_size,
                                     outDesc.get(),
                                     out.data()));
    } else {
        DiopiTensor out_temp = requiresTensor(ctx, out.shape(), input.dtype());
        CnnlTensorDesc out_tempDesc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                     matmul_desc,
                                     algo,
                                     &alpha,
                                     inputDesc.get(),
                                     input.data(),
                                     otherDesc.get(),
                                     other.data(),
                                     &beta,
                                     out_tempDesc.get(),
                                     out_temp.data(),
                                     workspace,
                                     workspace_size,
                                     out_tempDesc.get(),
                                     out_temp.data()));
        DIOPI_CALL(dataTypeCast(ctx, out, out_temp));
    }

    return diopiSuccess;
}

static diopiError_t matMulVector(diopiContextHandle_t ctx, DiopiTensor out_tensor, DiopiTensor input_tensor, DiopiTensor vector_tensor) {
    if (input_tensor.shape()[1] != vector_tensor.shape()[0]) {
        vector_tensor.reshape({1, vector_tensor.shape()[0]});
        out_tensor.reshape({vector_tensor.shape()[0], 1});
    } else {
        vector_tensor.reshape({vector_tensor.shape()[0], 1});
        out_tensor.reshape({input_tensor.shape()[0], 1});
    }

    DIOPI_CALL(matMulMat(ctx, out_tensor, input_tensor, vector_tensor));
    return diopiSuccess;
}

static diopiError_t transpose(diopiContextHandle_t ctx, DiopiTensor out_tensor, DiopiTensor input, int64_t dim0, int64_t dim1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiTensorHandle_t out = (diopiTensorHandle_t)out_tensor;

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> CnnlTransposeDesc;
    cnnlTransposeDescriptor_t transpose_desc = CnnlTransposeDesc.get();
    std::vector<int> perms = getPerm(input, dim0, dim1);
    cnnlSetTransposeDescriptor(transpose_desc, perms.size(), perms.data());

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transpose_desc, &workspace_size);
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    cnnlTranspose_v2(handle, transpose_desc, inputDesc.get(), input.data(), outDesc.get(), out_tensor.data(), workspace, workspace_size);
    return diopiSuccess;
}

static diopiError_t batchMatmul(diopiContextHandle_t ctx, DiopiTensor out_tensor, DiopiTensor input_tensor, DiopiTensor other_tensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, other_tensor, diopi_dtype_float32));
    }

    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(other_tensor, CNNL_LAYOUT_ARRAY);

    int32_t allow_tf32_int = 1;
    CnnlDescBase<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> bmmDesc;
    cnnlSetMatMulDescAttr(bmmDesc.get(), CNNL_MATMUL_ALLOW_TF32, &allow_tf32_int, sizeof(allow_tf32_int));
    CnnlDescBase<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> bmmAlgo;
    CnnlDescBase<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> bmmHeuristicResult;

    int return_algo_count = 0;
    cnnlGetBatchMatMulAlgoHeuristic(
        handle, bmmDesc.get(), inputDesc.get(), otherDesc.get(), outDesc.get(), nullptr, 1, &(bmmHeuristicResult.get()), &return_algo_count);

    size_t workspace_size(0);
    cnnlGetBatchMatMulHeuristicResult(bmmHeuristicResult.get(), bmmAlgo.get(), &workspace_size);
    void* workspace = nullptr;
    if (workspace > 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    if (out_tensor.dtype() == input_tensor.dtype()) {
        DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
                                               bmmDesc.get(),
                                               bmmAlgo.get(),
                                               nullptr,
                                               inputDesc.get(),
                                               input_tensor.data(),
                                               otherDesc.get(),
                                               other_tensor.data(),
                                               nullptr,
                                               outDesc.get(),
                                               out_tensor.data(),
                                               workspace,
                                               workspace_size));
    } else {
        DiopiTensor out_temp = requiresTensor(ctx, out_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_tempDesc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
                                               bmmDesc.get(),
                                               bmmAlgo.get(),
                                               nullptr,
                                               inputDesc.get(),
                                               input_tensor.data(),
                                               otherDesc.get(),
                                               other_tensor.data(),
                                               nullptr,
                                               out_tempDesc.get(),
                                               out_temp.data(),
                                               workspace,
                                               workspace_size));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp));
    }

    return diopiSuccess;
}

static diopiError_t tensorMatmulTensor(diopiContextHandle_t ctx, DiopiTensor out_tensor, DiopiTensor input_tensor, DiopiTensor other_tensor) {
    if (input_tensor.dim() == 1 && other_tensor.dim() == 1) {
        DIOPI_CALL(vectorMulVector(ctx, out_tensor, input_tensor, other_tensor));
        return diopiSuccess;
    } else if (input_tensor.dim() == 2 && other_tensor.dim() == 1) {
        DIOPI_CALL(matMulVector(ctx, out_tensor, input_tensor, other_tensor));
        return diopiSuccess;
    } else if (input_tensor.dim() == 1 && other_tensor.dim() == 2) {
        std::vector<int64_t> shape(other_tensor.shape());
        shape[0] = other_tensor.shape()[1];
        shape[1] = other_tensor.shape()[0];
        DiopiTensor other_T = requiresTensor(ctx, shape, other_tensor.dtype());
        DIOPI_CALL(transpose(ctx, other_T, other_tensor, 0, 1))
        DIOPI_CALL(matMulVector(ctx, out_tensor, other_T, input_tensor));
        return diopiSuccess;
    } else if (input_tensor.dim() == 2 && other_tensor.dim() == 2) {
        DIOPI_CALL(matMulMat(ctx, out_tensor, input_tensor, other_tensor));
        return diopiSuccess;
    } else if (input_tensor.dim() >= 3 && (other_tensor.dim() == 1 || other_tensor.dim() == 2)) {
        std::vector<int64_t> output_size;
        output_size.insert(output_size.end(), input_tensor.shape().begin(), input_tensor.shape().end() - 1);
        if (other_tensor.dim() == 1) {
            std::vector<int64_t> temp_shape(2);
            temp_shape[0] = other_tensor.shape()[0];
            temp_shape[1] = 1;
            other_tensor.reshape(temp_shape);
        } else {
            output_size.push_back(other_tensor.shape()[1]);
        }

        std::vector<int64_t> shape(2);
        shape[1] = input_tensor.shape()[input_tensor.dim() - 1];
        shape[0] = input_tensor.numel() / shape[1];
        input_tensor.reshape(shape);
        shape[1] = other_tensor.shape()[1];
        out_tensor.reshape(shape);
        DIOPI_CALL(matMulMat(ctx, out_tensor, input_tensor, other_tensor));
        return diopiSuccess;
    } else if ((input_tensor.dim() == 1 || input_tensor.dim() == 2) && other_tensor.dim() >= 3) {
        int input_dim = input_tensor.dim();
        int64_t n = input_tensor.dim() == 2 ? input_tensor.shape()[0] : 1;
        int64_t m = input_tensor.shape()[input_tensor.dim() - 1];
        int64_t p = other_tensor.shape()[other_tensor.dim() - 1];
        if (input_dim == 1) {
            input_tensor.reshape({n, m});
        }

        std::vector<int64_t> other_shape(other_tensor.shape());
        other_shape[other_tensor.shape().size() - 1] = other_tensor.shape()[other_tensor.shape().size() - 2];
        other_shape[other_tensor.shape().size() - 2] = other_tensor.shape()[other_tensor.shape().size() - 1];
        DiopiTensor other_T_tensor = requiresTensor(ctx, other_shape, other_tensor.dtype());
        DIOPI_CALL(transpose(ctx, other_T_tensor, other_tensor, -1, -2))
        std::vector<int64_t> input_shape(input_tensor.shape());
        input_shape[0] = input_tensor.shape()[1];
        input_shape[1] = input_tensor.shape()[0];
        DiopiTensor input_T_tensor = requiresTensor(ctx, input_shape, input_tensor.dtype());
        DIOPI_CALL(transpose(ctx, input_T_tensor, input_tensor, 0, 1))

        if (input_dim == 1) {
            DIOPI_CALL(tensorMatmulTensor(ctx, out_tensor, other_T_tensor, input_T_tensor));
        } else {
            std::vector<int64_t> shape(other_T_tensor.shape().begin(), other_T_tensor.shape().end() - 1);
            shape.push_back(input_tensor.shape()[0]);
            DiopiTensor out_temp = requiresTensor(ctx, shape, out_tensor.dtype());

            DIOPI_CALL(tensorMatmulTensor(ctx, out_temp, other_T_tensor, input_T_tensor));
            DIOPI_CALL(transpose(ctx, out_tensor, out_temp, -1, -2));
        }

        return diopiSuccess;
    } else if ((input_tensor.dim() >= 1 && other_tensor.dim() >= 1) && (input_tensor.dim() >= 3 || other_tensor.dim() >= 3)) {
        int64_t n = input_tensor.dim() > 1 ? input_tensor.shape()[input_tensor.dim() - 2] : 1;
        int64_t m1 = input_tensor.shape()[input_tensor.dim() - 1];
        int64_t data_len = input_tensor.dim() > 2 ? input_tensor.shape().size() - 2 : 0;
        std::vector<int64_t> batch_tensor1(input_tensor.shape().begin(), input_tensor.shape().begin() + data_len);

        int64_t m2 = other_tensor.dim() > 1 ? other_tensor.shape()[input_tensor.dim() - 2] : 1;
        int64_t p = other_tensor.shape()[other_tensor.dim() - 1];
        data_len = other_tensor.dim() > 2 ? other_tensor.shape().size() - 2 : 0;
        std::vector<int64_t> batch_tensor2(other_tensor.shape().begin(), other_tensor.shape().begin() + data_len);

        std::vector<int64_t> expand_batch_portion = inferSize(batch_tensor1, batch_tensor2);
        std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
        tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});
        std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
        tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

        int64_t expand_batch_product = multiplyIntegers(expand_batch_portion);
        std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
        tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});
        std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
        tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

        DiopiTensor input_expand = requiresTensor(ctx, tensor1_expand_size, input_tensor.dtype());
        DiopiTensor other_expand = requiresTensor(ctx, tensor2_expand_size, other_tensor.dtype());
        broadcast(ctx, input_expand, input_tensor);
        broadcast(ctx, other_expand, other_tensor);
        input_expand.reshape(tensor1_bmm_view);
        other_expand.reshape(tensor2_bmm_view);

        std::vector<int64_t> output_shape({expand_batch_product});
        if (input_tensor.dim() > 1) {
            output_shape.push_back(n);
        }
        if (other_tensor.dim() > 1) {
            output_shape.push_back(p);
        }
        out_tensor.reshape(output_shape);
        DIOPI_CALL(batchMatmul(ctx, out_tensor, input_expand, other_expand));
        return diopiSuccess;
    }

    set_last_error_string("both arguments to matmul need to be at least 1D, but they are ", input_tensor.dim(), "D and ", other_tensor.dim(), "D");
    return diopiErrorOccurred;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor out_tensor(out);

    DIOPI_CALL(tensorMatmulTensor(ctx, out_tensor, input_tensor, other_tensor));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
