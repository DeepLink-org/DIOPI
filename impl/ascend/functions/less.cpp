/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    AclOpRunner<2, 1>("Less", ctx).addInput(input, other).addOutput(out).run();
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiLt(ctx, out, input, trOther);
}

}  // namespace ascend
}  // namespace impl
