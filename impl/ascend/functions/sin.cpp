/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    AclTensor inputAcl(input);
    if (!inputAcl.defined() || inputAcl.numel() == 0) {
        return diopiSuccess;
    }

    ACLNN_ADAPTOR(aclnnInplaceSin, ctx, inputAcl);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclTensor inputAcl(input), outAcl(out);
    if (!inputAcl.defined() || inputAcl.numel() == 0) {
        return diopiSuccess;
    }

    ACLNN_ADAPTOR(aclnnSin, ctx, inputAcl, outAcl);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
