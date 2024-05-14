/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                            int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnEmbedding, ctx, weight, indices, out);
    return diopiSuccess;
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                    int64_t numWeights, int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnEmbeddingDenseBackward, ctx, grad, indices, numWeights, paddingIdx, scaleGradByfreq, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
