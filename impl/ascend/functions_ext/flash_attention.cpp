/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/aclnn.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                                 diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal) {
    aclnnFlashAttentionAdaptor(ctx, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, q, k, v, pDropout, softmaxScale, isCausal);
    return diopiSuccess;
}

diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                         diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                         diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                         diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal) {
    aclnnFlashAttentionBackwardAdaptor(
        ctx, gradQ, gradK, gradV, gradOut, q, k, v, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, pDropout, softmaxScale, isCausal);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
