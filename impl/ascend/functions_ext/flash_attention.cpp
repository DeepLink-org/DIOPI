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

diopiError_t diopiCustomizedFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* dropoutMask,
                                           diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum, diopiTensorHandle_t* softmaxOut,
                                           diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                           diopiConstTensorHandle_t alibiSlopes, diopiConstTensorHandle_t attentionMask, float pDropout, float softmaxScale,
                                           bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight, int32_t headNum, const char* inputLayout) {
    AscendTensor qAt(q), kAt(k), vAt(v), alibiSlopesAt(alibiSlopes), attentionMaskAt(attentionMask), attentionOutAt(attentionOut);
    ASCEND_CHECK_ABORT(alibiSlopes == nullptr, "For ascend, flash attention currently does not support Attention with Linear Biases (ALiBi)!");
    ASCEND_CHECK_ABORT(windowSizeLeft == -1 && windowSizeRight == -1, "For ascend, flash attention currently does not support sliding window local attention!");
    ASCEND_CHECK_ABORT(qAt.dim() == 3 || qAt.dim() == 4, "The shapes of the input query should be 3-dimensional or 4-dimensional");
    ASCEND_CHECK_ABORT(kAt.dim() == 3 || kAt.dim() == 4, "The shapes of the input key should be 3-dimensional or 4-dimensional");
    ASCEND_CHECK_ABORT(vAt.dim() == 3 || vAt.dim() == 4, "The shapes of the input value should be 3-dimensional or 4-dimensional");
    ASCEND_CHECK_ABORT(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");
    ASCEND_CHECK_ABORT(isCausal == false || attentionMask != nullptr, "When isCausal is True, attentionMask should not be nullptr!");

    int64_t b = 0;
    int64_t s0 = 0;
    int64_t s1 = 0;
    int64_t n = 0;
    int64_t d = 0;
    int64_t h = 0;

    if (strcmp(inputLayout, "SBH") == 0) {
        b = qAt.shape(1);
        s0 = qAt.shape(0);  // S for query
        s1 = kAt.shape(0);  // S for key & value
        n = headNum;
        h = qAt.shape(2);
    } else if (strcmp(inputLayout, "BSH") == 0) {
        b = qAt.shape(0);
        s0 = qAt.shape(1);  // S for query
        s1 = kAt.shape(1);  // S for key & value
        n = headNum;
        h = qAt.shape(2);
    } else if (strcmp(inputLayout, "BSND") == 0) {
        b = qAt.shape(0);
        s0 = qAt.shape(1);  // S for query
        s1 = kAt.shape(1);  // S for key & value
        n = qAt.shape(2);
        d = qAt.shape(3);
    } else if (strcmp(inputLayout, "BNSD") == 0) {
        b = qAt.shape(0);
        s0 = qAt.shape(2);  // S for query
        s1 = kAt.shape(2);  // S for key & value
        n = qAt.shape(1);
        d = qAt.shape(3);
    } else {
        ASCEND_CHECK_ABORT(false, "The input layout should be BSH/SBH/BNSD/BSND");
    }

    double keepProb = static_cast<double>(1 - pDropout);

    AscendTensor pseAt, paddingMaskAt, dropoutMaskAt;
    diopiSize_t prefixN{nullptr, 0};
    if (pDropout > 0 && pDropout <= 1) {
        int64_t numels = b * n * s0 * s1;  // [B,N,S,S]
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

    int64_t preTokens = kAt.shape(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;
    if (isCausal) {
        if (s0 > 2048 && s1 > 2048 && attentionMaskAt.defined() && attentionMaskAt.shape(0) == 2048 && attentionMaskAt.shape(1) == 2048) {
            sparseMode = 2;
        }
    }

    AscendTensor softmaxMaxAt, softmaxSumAt, softmaxOutAt;

    makeTensor(ctx, softmaxMaxAt, {b, n, s0, 8}, diopi_dtype_float32);
    makeTensor(ctx, softmaxSumAt, {b, n, s0, 8}, diopi_dtype_float32);
    makeTensor(ctx, softmaxOutAt, std::vector<int64_t>{0}, qAt.dtype());
    double scale = static_cast<double>(softmaxScale);
    DIOPI_ASCEND_CALL_ACLNN(aclnnFlashAttentionScore,
                            ctx,
                            qAt,
                            kAt,
                            vAt,
                            pseAt,
                            dropoutMaskAt,
                            paddingMaskAt,
                            attentionMaskAt,
                            prefixN,
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

diopiError_t diopiCustomizedFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                                   diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                                   diopiConstTensorHandle_t v, diopiConstTensorHandle_t alibiSlopes, diopiConstTensorHandle_t attentionOut,
                                                   diopiConstTensorHandle_t attentionMask, diopiConstTensorHandle_t dropoutMask,
                                                   diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                                   diopiConstTensorHandle_t softmaxOut, float pDropout, float softmaxScale, bool isCausal,
                                                   int32_t windowSizeLeft, int32_t windowSizeRight, int32_t headNum, const char* inputLayout) {
    AscendTensor qAt(q), kAt(k), vAt(v), attentionOutAt(attentionOut), attentionMaskAt(attentionMask), softmaxMaxAt(softmaxMax), softmaxSumAt(softmaxSum),
        softmaxOutAt(softmaxOut), gradQAt(gradQ), gradKAt(gradK), gradVAt(gradV), gradOutAt(gradOut);
    ASCEND_CHECK_ABORT(alibiSlopes == nullptr, "For ascend, flash attention currently does not support Attention with Linear Biases (ALiBi)!");
    ASCEND_CHECK_ABORT(windowSizeLeft == -1 && windowSizeRight == -1, "For ascend, flash attention currently does not support sliding window local attention!");
    ASCEND_CHECK_ABORT(qAt.dim() == 3 || qAt.dim() == 4, "The shapes of the input query should be 3-dimensional or 4-dimensional");
    ASCEND_CHECK_ABORT(kAt.dim() == 3 || kAt.dim() == 4, "The shapes of the input key should be 3-dimensional or 4-dimensional");
    ASCEND_CHECK_ABORT(vAt.dim() == 3 || vAt.dim() == 4, "The shapes of the input value should be 3-dimensional or 4-dimensional");
    ASCEND_CHECK_ABORT(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");
    ASCEND_CHECK_ABORT(isCausal == false || attentionMask != nullptr, "When isCausal is True, attentionMask should not be nullptr!");

    AscendTensor dropoutMaskAt;
    if (dropoutMask) {
        dropoutMaskAt = AscendTensor(dropoutMask);
    }

    int64_t b = 0;
    int64_t s0 = 0;
    int64_t s1 = 0;
    int64_t n = 0;
    int64_t d = 0;
    int64_t h = 0;

    if (strcmp(inputLayout, "SBH") == 0) {
        b = qAt.shape(1);
        s0 = qAt.shape(0);  // S for query
        s1 = kAt.shape(0);  // S for key & value
        n = headNum;
        h = qAt.shape(2);
    } else if (strcmp(inputLayout, "BSH") == 0) {
        b = qAt.shape(0);
        s0 = qAt.shape(1);  // S for query
        s1 = kAt.shape(1);  // S for key & value
        n = headNum;
        h = qAt.shape(2);
    } else if (strcmp(inputLayout, "BSND") == 0) {
        b = qAt.shape(0);
        s0 = qAt.shape(1);  // S for query
        s1 = kAt.shape(1);  // S for key & value
        n = qAt.shape(2);
        d = qAt.shape(3);
    } else if (strcmp(inputLayout, "BNSD") == 0) {
        b = qAt.shape(0);
        s0 = qAt.shape(2);  // S for query
        s1 = kAt.shape(2);  // S for key & value
        n = qAt.shape(1);
        d = qAt.shape(3);
    } else {
        ASCEND_CHECK_ABORT(false, "The input layout should be BSH/SBH/BNSD/BSND");
    }

    double keepProb = static_cast<double>(1 - pDropout);

    AscendTensor pseAt;
    AscendTensor gradPseAt;
    makeTensor(ctx, gradPseAt, std::vector<int64_t>{0}, qAt.dtype());
    diopiSize_t prefixN{nullptr, 0};
    AscendTensor paddingMaskAt;
    int64_t preTokens = kAt.shape(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;
    if (isCausal) {
        if (s0 > 2048 && s1 > 2048 && attentionMaskAt.defined() && attentionMaskAt.shape(0) == 2048 && attentionMaskAt.shape(1) == 2048) {
            sparseMode = 2;
        }
    }

    double scale = static_cast<double>(softmaxScale);
    DIOPI_ASCEND_CALL_ACLNN(aclnnFlashAttentionScoreGrad,
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
