/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Cast", ctx).addInput(input).addOutput(out).setAttr<int32_t>("dst_type", getAclDataType(out)).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
