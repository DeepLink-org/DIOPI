/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t
diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    auto out_tensor = DiopiTensor(out);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    float min = min_val->fval;
    float max = max_val->fval;
    if (min > max) {
        min = max;
    }

    DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), max, min, outDesc.get(), out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);

    float min = min_val->fval;
    float max = max_val->fval;
    if (min > max) {
        min = max;
    }

    DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), max, min, inputDesc.get(), input_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t grad_input,
                                             diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input,
                                             const diopiScalar_t* min_val,
                                             const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    auto grad_out_tensor = DiopiTensor(grad_output);
    CnnlTensorDesc gradoutDesc(grad_out_tensor, CNNL_LAYOUT_ARRAY);
    auto grad_in_tensor = DiopiTensor(grad_input);
    CnnlTensorDesc gradinDesc(grad_in_tensor, CNNL_LAYOUT_ARRAY);

    float min = min_val->fval;
    float max = max_val->fval;
    if (min > max) {
        min = max;
    }

    DIOPI_CALLCNNL(cnnlHardtanhBackward(
        handle, inputDesc.get(), input_tensor.data(), gradoutDesc.get(), grad_out_tensor.data(), max, min, gradinDesc.get(), grad_in_tensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
