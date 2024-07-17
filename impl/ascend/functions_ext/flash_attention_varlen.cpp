/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

namespace {

const int64_t bitNumber = 128;
const int64_t uInt8BitNumber = 8;

}  // namespace

diopiError_t diopiCustomizedFlashAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* dropoutMask,
                                                 diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum, diopiTensorHandle_t* softmaxOut,
                                                 diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                 diopiSize_t cumSeqQ, diopiSize_t cumSeqKV, diopiConstTensorHandle_t alibiSlopes,
                                                 diopiConstTensorHandle_t attentionMask, int32_t maxSeqLenQ, int32_t maxSeqLenKV, float pDropout,
                                                 float softmaxScale, bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    AscendTensor qAt(q), kAt(k), vAt(v), attentionMaskAt(attentionMask), attentionOutAt(attentionOut);
    ASCEND_CHECK_ABORT(alibiSlopes == nullptr, "For ascend, flash attention currently does not support Attention with Linear Biases (ALiBi)!");
    ASCEND_CHECK_ABORT(windowSizeLeft == -1 && windowSizeRight == -1, "For ascend, flash attention currently does not support sliding window local attention!");
    ASCEND_CHECK_ABORT(qAt.dim() == 3, "The shapes of the input query should be 3-dimensional");
    ASCEND_CHECK_ABORT(kAt.dim() == 3, "The shapes of the input key should be 3-dimensional");
    ASCEND_CHECK_ABORT(vAt.dim() == 3, "The shapes of the input value should be 3-dimensional");
    ASCEND_CHECK_ABORT(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");
    ASCEND_CHECK_ABORT(isCausal == false || attentionMask != nullptr, "When isCausal is True, attentionMask should not be nullptr!");

    const char* inputLayout = "TND";

    int64_t t = qAt.shape(0);
    int64_t n = qAt.shape(1);
    int64_t d = qAt.shape(2);

    double keepProb = static_cast<double>(1 - pDropout);

    AscendTensor pseAt, paddingMaskAt, dropoutMaskAt;
    diopiSize_t prefixN{nullptr, 0};
    if (pDropout > 0 && pDropout <= 1) {
        int64_t numels = n;
        int64_t accum = cumSeqQ.data[0] * cumSeqKV.data[0];
        for (int64_t i = 1; i < cumSeqQ.len; i++) {
            accum += ((cumSeqQ.data[i] - cumSeqQ.data[i - 1]) * (cumSeqKV.data[i] - cumSeqKV.data[i - 1]));
        }
        int64_t length = (numels + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        makeTensor(ctx, dropoutMaskAt, {length}, diopi_dtype_uint8);
        if (pDropout == 1) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, dropoutMaskAt);
        } else {
            std::vector<int64_t> shapeVector{numels};
            const std::pair<uint64_t, uint64_t> pair = getSeedAndOffset(ctx, gen, 10);
            const uint64_t seed = pair.first;
            const uint64_t offset = pair.second;
            DIOPI_ASCEND_CALL_ACLNN(aclnnDropoutGenMask, ctx, shapeVector, pDropout, seed, offset, dropoutMaskAt);
        }
    }

    int64_t preTokens = kAt.shape(0);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;
    if (isCausal) {
        if (maxSeqLenQ > 2048 && maxSeqLenKV > 2048 && attentionMaskAt.defined() && attentionMaskAt.shape(0) == 2048 && attentionMaskAt.shape(1) == 2048) {
            sparseMode = 2;
        }
    }
    AscendTensor softmaxMaxAt, softmaxSumAt, softmaxOutAt;

    makeTensor(ctx, softmaxMaxAt, {t, n, 8}, diopi_dtype_float32);  // [T, N, 8]
    makeTensor(ctx, softmaxSumAt, {t, n, 8}, diopi_dtype_float32);  // [T, N, 8]
    makeTensor(ctx, softmaxOutAt, std::vector<int64_t>{0}, qAt.dtype());
    double scale = static_cast<double>(softmaxScale);
    DIOPI_ASCEND_CALL_ACLNN(aclnnFlashAttentionVarLenScore,
                            ctx,
                            qAt,
                            kAt,
                            vAt,
                            pseAt,
                            dropoutMaskAt,
                            paddingMaskAt,
                            attentionMaskAt,
                            prefixN,
                            cumSeqQ,
                            cumSeqKV,
                            scale,
                            keepProb,
                            preTokens,
                            nextTokens,
                            n,
                            inputLayout,
                            innerPrecise,
                            sparseMode,
                            softmaxMaxAt,
                            softmaxSumAt,
                            softmaxOutAt,
                            attentionOutAt);

    if (dropoutMaskAt.defined()) {
        *dropoutMask = diopiTensorHandle_t(dropoutMaskAt);
    }
    *softmaxMax = diopiTensorHandle_t(softmaxMaxAt);
    *softmaxSum = diopiTensorHandle_t(softmaxSumAt);
    *softmaxOut = diopiTensorHandle_t(softmaxOutAt);
    return diopiSuccess;
}

diopiError_t diopiCustomizedFlashAttentionVarLenBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK,
                                                         diopiTensorHandle_t gradV, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q,
                                                         diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiSize_t cumSeqQ, diopiSize_t cumSeqKV,
                                                         diopiConstTensorHandle_t alibiSlopes, diopiConstTensorHandle_t attentionOut,
                                                         diopiConstTensorHandle_t attentionMask, diopiConstTensorHandle_t dropoutMask,
                                                         diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                                         diopiConstTensorHandle_t softmaxOut, int32_t maxSeqLenQ, int32_t maxSeqLenKV, float pDropout,
                                                         float softmaxScale, bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    AscendTensor qAt(q), kAt(k), vAt(v), attentionOutAt(attentionOut), attentionMaskAt(attentionMask), softmaxMaxAt(softmaxMax), softmaxSumAt(softmaxSum),
        softmaxOutAt(softmaxOut), gradQAt(gradQ), gradKAt(gradK), gradVAt(gradV), gradOutAt(gradOut);
    ASCEND_CHECK_ABORT(alibiSlopes == nullptr, "For ascend, flash attention currently does not support Attention with Linear Biases (ALiBi)!");
    ASCEND_CHECK_ABORT(windowSizeLeft == -1 && windowSizeRight == -1, "For ascend, flash attention currently does not support sliding window local attention!");
    ASCEND_CHECK_ABORT(qAt.dim() == 3, "The shapes of the input query should be 3-dimensional");
    ASCEND_CHECK_ABORT(kAt.dim() == 3, "The shapes of the input key should be 3-dimensional");
    ASCEND_CHECK_ABORT(vAt.dim() == 3, "The shapes of the input value should be 3-dimensional");
    ASCEND_CHECK_ABORT(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");
    ASCEND_CHECK_ABORT(isCausal == false || attentionMask != nullptr, "When isCausal is True, attentionMask should not be nullptr!");

    const char* inputLayout = "TND";

    int64_t t = qAt.shape(0);
    int64_t n = qAt.shape(1);
    int64_t d = qAt.shape(2);

    double keepProb = static_cast<double>(1 - pDropout);

    AscendTensor pseAt;
    AscendTensor gradPseAt;
    makeTensor(ctx, gradPseAt, std::vector<int64_t>{0}, qAt.dtype());
    diopiSize_t prefixN{nullptr, 0};
    AscendTensor paddingMaskAt;

    int64_t preTokens = kAt.shape(0);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;
    if (isCausal) {
        if (maxSeqLenQ > 2048 && maxSeqLenKV > 2048 && attentionMaskAt.defined() && attentionMaskAt.shape(0) == 2048 && attentionMaskAt.shape(1) == 2048) {
            sparseMode = 2;
        }
    }

    AscendTensor dropoutMaskAt;
    if (dropoutMask) {
        dropoutMaskAt = AscendTensor(dropoutMask);
    }

    double scale = static_cast<double>(softmaxScale);
    DIOPI_ASCEND_CALL_ACLNN(aclnnFlashAttentionUnpaddingScoreGrad,
                            ctx,
                            qAt,
                            kAt,
                            vAt,
                            gradOutAt,
                            pseAt,
                            dropoutMaskAt,
                            paddingMaskAt,
                            attentionMaskAt,
                            softmaxMaxAt,
                            softmaxSumAt,
                            softmaxOutAt,
                            attentionOutAt,
                            prefixN,
                            cumSeqQ,
                            cumSeqKV,
                            scale,
                            keepProb,
                            preTokens,
                            nextTokens,
                            n,
                            inputLayout,
                            innerPrecise,
                            sparseMode,
                            gradQAt,
                            gradKAt,
                            gradVAt,
                            gradPseAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
