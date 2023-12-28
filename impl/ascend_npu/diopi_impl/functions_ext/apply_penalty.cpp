/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                               diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t pTokenIds, diopiConstTensorHandle_t pTokenCounts,
                               diopiConstTensorHandle_t pCumsumSeqLen, int pMaxLenInBatch) {
    BEGIN_CALL_ACL_OP(logits, presencePenalty, frequencyPenalty, pTokenIds, pTokenCounts, pCumsumSeqLen);
    int batch = logitsAt.size(0);
    c10::Device device = logitsAt.device();
    c10::Layout layout = logitsAt.layout();
    for (int i = 0; i < batch; ++i) {
        int curBatchStartIndex = pCumsumSeqLenAt[i].item<int>();
        int curBatchEndIndex = pCumsumSeqLenAt[i + 1].item<int>();
        at::Tensor slice = acl_op::arange(curBatchStartIndex, curBatchEndIndex, at::kLong, layout, device);
        at::Tensor curTokenIds = at::index(pTokenIdsAt, {slice});
        at::Tensor curTokenCounts = at::index(pTokenCountsAt, {slice});
        at::Tensor curLogits = logitsAt[i].index_select(0, curTokenIds);
        curLogits = curLogits - curTokenCounts * frequencyPenaltyAt[i] - presencePenaltyAt[i];
        at::index_put_(logitsAt, {torch::scalar_to_tensor(i), curTokenIds}, curLogits);
    }
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
