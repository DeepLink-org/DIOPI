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
        AclTensor inputAcl(input), outAcl(out);
        if (!inputAcl.defined() || inputAcl.numel() == 0) {
            return diopiSuccess;
        }

        ACLNN_ADAPTOR(aclnnTriu, ctx, inputAcl, diagonal, outAcl);
        return diopiSuccess;
    } else {
        AclOpRunner<1, 1>("Triu", ctx).addInput(input).setAttr("diagonal", diagonal).addOutput(out).run();
        return diopiSuccess;
    }
}

diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    if (useAclnn()) {
        AclTensor inputAcl(input);
        if (!inputAcl.defined() || inputAcl.numel() == 0) {
            return diopiSuccess;
        }

        ACLNN_ADAPTOR(aclnnInplaceTriu, ctx, inputAcl, diagonal);
        return diopiSuccess;
    } else {
        return diopiTriu(ctx, input, input, diagonal);
    }
}

}  // namespace ascend
}  // namespace impl
