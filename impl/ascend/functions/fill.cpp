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

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    AclOpRunner<1, 1>("Fills", ctx).addInput(input).setAttr<float>("value", getValue<float>(value)).addOutput(input).run();
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
