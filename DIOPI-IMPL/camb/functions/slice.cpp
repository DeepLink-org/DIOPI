/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    DiopiTensor index_tensor(index);
    DiopiTensor out_tensor(out);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + input_tensor.dim();
    }
    if (out_tensor.dtype() == input_tensor.dtype()) {
        DIOPI_CALLCNNL(
            cnnlIndexSelect(handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), outDesc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp_tensor = requiresTensor(ctx, out_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_tempDesc(out_temp_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlIndexSelect(
            handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), out_tempDesc.get(), out_temp_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp_tensor));
    }
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_int64, 0};
    DIOPI_CALL(diopiFill(ctx, grad_input, &zero));
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_tensor(grad);
    DiopiTensor out_tensor(grad_input);
    diopiDtype_t out_dtype = grad_input_tensor.dtype();
    if (grad_input_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, diopi_dtype_int32));
    } else if (grad_input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, diopi_dtype_float32));
    }
    if (grad_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_tensor, diopi_dtype_int32));
    } else if (grad_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc grad_inputDesc(grad_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(grad_tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor index_tensor(index);
    if (index_tensor.dtype() != diopi_dtype_int32) {
        DIOPI_CALL(dataTypeCast(ctx, index_tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + input_sizes.len;
    }

    if (grad_input_tensor.dtype() == out_dtype) {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data(),
                                    indexDesc.get(),
                                    index_tensor.data(),
                                    gradDesc.get(),
                                    grad_tensor.data(),
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data()));
    } else {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data(),
                                    indexDesc.get(),
                                    index_tensor.data(),
                                    gradDesc.get(),
                                    grad_tensor.data(),
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, grad_input_tensor));
    }
    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }

    diopiScalar_t index_scalar;
    index_scalar.stype = diopi_dtype_int64;
    index_scalar.ival = index;
    DiopiTensor index_tensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, &index_scalar, index_tensor));
    DiopiTensor out_tensor(out);

    if (dim < 0) {
        dim = dim + input_tensor.dim();
    }
    std::vector<int64_t> shape(out_tensor.shape());
    shape.insert(shape.begin() + dim, 1);
    out_tensor.reshape(shape);

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    if (out_tensor.dtype() == input_tensor.dtype()) {
        DIOPI_CALLCNNL(
            cnnlIndexSelect(handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), outDesc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp_tensor = requiresTensor(ctx, out_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_tempDesc(out_temp_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlIndexSelect(
            handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), out_tempDesc.get(), out_temp_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp_tensor));
    }
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                 int64_t dim, int64_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_int64, 0};
    DIOPI_CALL(diopiFill(ctx, grad_input, &zero));
    DiopiTensor grad_input_tensor(grad_input);
    diopiDtype_t out_dtype = grad_input_tensor.dtype();
    if (dim < 0) {
        dim = dim + input_sizes.len;
    }
    DiopiTensor grad_tensor(grad_output);
    std::vector<int64_t> shape(grad_tensor.shape());
    shape.insert(shape.begin() + dim, 1);
    grad_tensor.reshape(shape);

    if (grad_input_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, diopi_dtype_int32));
    } else if (grad_input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, diopi_dtype_float32));
    }
    if (grad_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_tensor, diopi_dtype_int32));
    } else if (grad_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc grad_inputDesc(grad_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(grad_tensor, CNNL_LAYOUT_ARRAY);

    diopiScalar_t index_scalar;
    index_scalar.stype = diopi_dtype_int64;
    index_scalar.ival = index;
    DiopiTensor index_tensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, &index_scalar, index_tensor));
    if (index_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, index_tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);

    if (grad_input_tensor.dtype() == out_dtype) {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data(),
                                    indexDesc.get(),
                                    index_tensor.data(),
                                    gradDesc.get(),
                                    grad_tensor.data(),
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data()));
    } else {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data(),
                                    indexDesc.get(),
                                    index_tensor.data(),
                                    gradDesc.get(),
                                    grad_tensor.data(),
                                    grad_inputDesc.get(),
                                    grad_input_tensor.data()));
        DiopiTensor out_tensor(grad_input);
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, grad_input_tensor));
    }

    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(null_out);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> start_32(input_tensor.dim(), 0);
    std::vector<int32_t> step_32(input_tensor.dim(), 1);
    std::vector<int32_t> end_32(input_tensor.shape().begin(), input_tensor.shape().end());
    start_32[dim] = start;
    step_32[dim] = step;
    end_32[dim] = end;

    DIOPI_CALLCNNL(
        cnnlStridedSlice(handle, inputDesc.get(), input_tensor.data(), start_32.data(), end_32.data(), step_32.data(), outDesc.get(), out_tensor.data()));
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(grad_output);
    DiopiTensor out_tensor(grad_input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> start_32(input_tensor.dim(), 0);
    std::vector<int32_t> step_32(input_tensor.dim(), 1);
    std::vector<int32_t> end_32(input_tensor.shape().begin(), input_tensor.shape().end());
    start_32[dim] = start;
    step_32[dim] = step;
    end_32[dim] = end;

    if (out_tensor.dtype() == input_tensor.dtype()) {
        DIOPI_CALLCNNL(cnnlStridedSliceBackward(
            handle, start_32.data(), end_32.data(), step_32.data(), inputDesc.get(), input_tensor.data(), outDesc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp_tensor = requiresTensor(ctx, out_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_tempDesc(out_temp_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlStridedSliceBackward(
            handle, start_32.data(), end_32.data(), step_32.data(), inputDesc.get(), input_tensor.data(), out_tempDesc.get(), out_temp_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp_tensor));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
