/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiCos(ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclTensor inAcl(input), outAcl(out);
    if (!inAcl.defined() || inAcl.numel() == 0) {
        return diopiSuccess;
    }
    ACLNN_ADAPTOR(aclnnCos, ctx, inAcl, outAcl);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
