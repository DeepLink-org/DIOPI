/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);

    AclOpRunner<4, 1>("OneHot", ctx)
        .addInput(input)
        .addConstInput<int>(static_cast<int>(numClasses))
        .addConstInput<float>(1.0)
        .addConstInput<float>(0.0)
        .setAttr<int>("axis", -1)
        .addOutput(out)
        .run();
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
