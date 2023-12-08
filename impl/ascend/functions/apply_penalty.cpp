/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presence_penalty,
                               diopiConstTensorHandle_t frequency_penalty, diopiConstTensorHandle_t p_token_ids, diopiConstTensorHandle_t p_token_counts,
                               diopiConstTensorHandle_t p_cumsum_seq_len, int p_max_len_in_batch) {
    AscendTensor asLogits(logits);
    AscendTensor asPresencePenalty(presence_penalty);
    AscendTensor asFrequencyPenalty(frequency_penalty);
    AscendTensor asPTokenIds(p_token_ids);
    AscendTensor asPTokenCounts(p_token_counts);
    AscendTensor asPcumsumSeqLen(p_cumsum_seq_len);

    int64_t = asLogits.shape(0);
    for (int i = 0; i < batch; ++i) {
        // int64_t curBatchStartIndex = asPCumsumSeqLen[i].item<int>();
        // int64_t curBatchEndIndex = asPCumsumSeqLen[i + 1].item<int>();
        
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
