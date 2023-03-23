/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc x_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc y_desc(output_tensor, CNNL_LAYOUT_ARRAY);
    const void* x_ptr = input_tensor.data();
    void* y_ptr = output_tensor.data();

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> CnnlActivation;
    cnnlActivationDescriptor_t activation_desc = CnnlActivation.get();
    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activation_desc, CNNL_ACTIVATION_RELU, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));
    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, x_desc.get(), x_ptr, NULL, y_desc.get(), y_ptr));
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activate_guard;
    cnnlActivationDescriptor_t activation_desc = activate_guard.get();
    DIOPI_CALLCNNL(
        cnnlSetActivationDescriptor_v4(activation_desc, CNNL_ACTIVATION_RELU, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 0.0, 0, 0.0, 0.0));

    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc desc;
    desc.set(input_tensor, CNNL_LAYOUT_ARRAY);
    void* ptr = input_tensor.data();
    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc, NULL, desc.get(), ptr, NULL, desc.get(), ptr));
}

}  // namespace camb
}  // namespace impl
