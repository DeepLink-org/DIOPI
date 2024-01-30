/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    if (useAclnn()) {
        AclTensor inAcl(input), outAcl(out);
        if (!inAcl.defined() || inAcl.numel() == 0) {
            return diopiSuccess;
        }
        ACLNN_ADAPTOR("aclnnTriu", ctx, inAcl, diagonal, outAcl);
    } else {
        AclOpRunner<1, 1>("Triu", ctx).addInput(input).setAttr("diagonal", diagonal).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) { return diopiTriu(ctx, input, input, diagonal); }

}  // namespace ascend
}  // namespace impl
