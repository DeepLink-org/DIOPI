/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);

    AclOpRunner<4, 1>("OneHot", ctx)
        .addInput(input)
        .addConstInput(numClasses, diopi_dtype_int32)
        .addConstInput(1.0, diopi_dtype_float32)
        .addConstInput(0.0, diopi_dtype_float32)
        .setAttr<int>("axis", -1)
        .addOutput(out)
        .run();
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
