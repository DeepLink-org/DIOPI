/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    if (dtype == diopi_dtype_float16) {
        dtype = diopi_dtype_float64;
    } else {
        dtype = diopi_dtype_float32;
    }
    makeTensorFromScalar(ctx, value, &trOther, dtype, diopiDevice_t::diopi_device);

    AclOpRunner<4, 1>("Addcdiv", ctx).addInput(input, dtype).addInput(tensor1, dtype).addInput(tensor2, dtype).addInput(trOther, dtype).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    return diopiAddcdiv(ctx, input, input, tensor1, tensor2, value);
}

}  // namespace ascend
}  // namespace impl
