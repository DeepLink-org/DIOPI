/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

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
        at::Tensor slice = op_api::arange(curBatchStartIndex, curBatchEndIndex, at::kLong, layout, device);
        at::Tensor curTokenIds = at::index(pTokenIdsAt, {slice});
        at::Tensor curTokenCounts = at::index(pTokenCountsAt, {slice});
        at::Tensor curLogits = logitsAt[i].index_select(0, curTokenIds);
        curLogits = curLogits - curTokenCounts * frequencyPenaltyAt[i] - presencePenaltyAt[i];
        at::index_put_(logitsAt, {torch::scalar_to_tensor(i), curTokenIds}, curLogits);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiApplyPenaltyV2(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                               diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t repetitionPenalty,
                               diopiConstTensorHandle_t pTokenIds, diopiConstTensorHandle_t pTokenCounts) {
    BEGIN_CALL_ACL_OP(logits, presencePenalty, frequencyPenalty, repetitionPenalty, pTokenIds, pTokenCounts);
    logitsAt = impl::aten::viewStorage(logitsAt, {logitsAt.numel()});
    at::Tensor curLogits = op_api::index_select(logitsAt, 0, pTokenIdsAt);
    at::Tensor repoLogits = at_npu::native::OpPreparation::apply_tensor_without_format(curLogits);
    at::Tensor zero = at_npu::native::OpPreparation::apply_tensor_without_format(curLogits);
    op_api::zero_(zero);
    at::Tensor cand = at_npu::native::OpPreparation::apply_tensor_without_format(curLogits);
    op_api::gt_out(curLogits, zero, cand);
    op_api::where_out(cand, curLogits / repetitionPenaltyAt, curLogits * repetitionPenaltyAt, repoLogits);
    repoLogits = repoLogits - pTokenCountsAt * frequencyPenaltyAt - presencePenaltyAt;
    std::vector<int64_t> shape(pTokenIdsAt.dim() + 1, 1);
    for (int64_t i = 0; i < pTokenIdsAt.dim(); i++) {
        shape[i] = pTokenIdsAt.size(i);
    }
    pTokenIdsAt = impl::aten::viewStorage(pTokenIdsAt, shape);
    EXEC_NPU_CMD(aclnnScatterNd, logitsAt, pTokenIdsAt, repoLogits, logitsAt);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
