/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = makeTensor(input);
    auto output_tensor = makeTensor(out);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        set_last_error_string("%s", "Unsupport datatype float64");
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc x_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc y_desc(output_tensor, CNNL_LAYOUT_ARRAY);
    const void* x_ptr = input_tensor.data();
    void* y_ptr = output_tensor.data();

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activate_guard;
    cnnlActivationDescriptor_t activation_desc = activate_guard.get();

    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activation_desc, CNNL_ACTIVATION_SIGMOID, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));

    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, x_desc.get(), x_ptr, NULL, y_desc.get(), y_ptr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = makeTensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc x_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    void* x_ptr = input_tensor.data();

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    cnnlActivationDescriptor_t activation_desc = CnnlActivation.get();
    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activation_desc, CNNL_ACTIVATION_SIGMOID, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));
    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, x_desc.get(), x_ptr, NULL, x_desc.get(), x_ptr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t grad_input,
                                             diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_grad = makeTensor(grad_input);
    auto output_grad = makeTensor(grad_output);
    auto output_tensor = makeTensor(output);

    CnnlTensorDesc input_grad_desc(input_grad, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_grad_desc(output_grad, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);

    const void* output_ptr = output_tensor.data();
    const void* output_grad_ptr = output_grad.data();
    void* input_grad_ptr = input_grad.data();

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    cnnlActivationDescriptor_t activation_desc = CnnlActivation.get();
    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activation_desc, CNNL_ACTIVATION_SIGMOID, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));

    DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                          activation_desc,
                                          NULL,
                                          output_desc.get(),
                                          output_ptr,
                                          output_grad_desc.get(),
                                          output_grad_ptr,
                                          input_grad_desc.get(),
                                          input_grad_ptr,
                                          NULL,
                                          input_grad_desc.get(),
                                          input_grad_ptr));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
