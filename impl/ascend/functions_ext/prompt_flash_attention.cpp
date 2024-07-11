/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiPromptFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t query, diopiConstTensorHandle_t key,
                                       diopiConstTensorHandle_t value, diopiConstTensorHandle_t attenMask, diopiSize_t actualSeqLengths, int64_t maxInputLen,
                                       int64_t numHeads, int64_t numKeyValueHeads, int64_t dim) {
    AscendTensor queryAt(query), outAt(out), keyAt(key), valueAt(value);
    if (queryAt.dim() == 2) {
        queryAt = queryAt.view({actualSeqLengths.len, maxInputLen, queryAt.shape(1)});
        outAt = outAt.view({actualSeqLengths.len, maxInputLen, outAt.shape(1)});
        keyAt = keyAt.view({actualSeqLengths.len, maxInputLen, keyAt.shape(1)});
        valueAt = valueAt.view({actualSeqLengths.len, maxInputLen, valueAt.shape(1)});
    }
    double scaleValue = 1 / std::sqrt(dim);
    int64_t preTokens = 2147473647;
    int64_t nextTokens = 0;
    AscendTensor paddingMask;
    DIOPI_ASCEND_CALL_ACLNN(aclnnPromptFlashAttention,
                            ctx,
                            queryAt,
                            keyAt,
                            valueAt,
                            paddingMask,
                            attenMask,
                            actualSeqLengths,
                            numHeads,
                            scaleValue,
                            preTokens,
                            nextTokens,
                            "BSH",
                            numKeyValueHeads,
                            outAt);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
