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
DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                   diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64 || input_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    DiopiTensor index_tensor(index);
    DIOPI_CALL(autoCastTensorType(ctx, {&index_tensor}, {diopi_dtype_int32, diopi_dtype_int64}));
    DiopiTensor out_tensor(out);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim += input_tensor.dim();
    }

    if (out_tensor.dtype() == input_tensor.dtype()) {
        DIOPI_CALLCNNL(cnnlGather(handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), outDesc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp = out_tensor;
        DIOPI_CALL(dataTypeCast(ctx, out_temp, input_tensor.dtype()));
        CnnlTensorDesc out_tempDesc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlGather(handle, dim, inputDesc.get(), input_tensor.data(), indexDesc.get(), index_tensor.data(), out_tempDesc.get(), out_temp.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_float32, 0};
    DIOPI_CALL(diopiFill(ctx, grad_input, &zero));

    DiopiTensor input_tensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    DiopiTensor index_tensor(index);
    DIOPI_CALL(autoCastTensorType(ctx, {&index_tensor}, {diopi_dtype_int32, diopi_dtype_int64}));
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor out_temp = grad_input_tensor;
    if (out_temp.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, out_temp, diopi_dtype_float32));
    }
    DiopiTensor grad_output_tensor(grad_output);
    if (grad_output_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_output_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_tempDesc(out_temp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_outputDesc(grad_output_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlScatter(handle,
                               dim,
                               out_tempDesc.get(),
                               out_temp.data(),
                               indexDesc.get(),
                               index_tensor.data(),
                               grad_outputDesc.get(),
                               grad_output_tensor.data(),
                               out_tempDesc.get(),
                               out_temp.data(),
                               CNNL_SCATTER_ADD));
    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, out_temp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
