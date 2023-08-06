/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {

DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Log", ctx).addInput(input).setAttr<float>("base", -1.0).setAttr<float>("scale", 1.0).setAttr<float>("shift", 0.0).addOutput(out).run();
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
