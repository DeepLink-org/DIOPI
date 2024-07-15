/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiPagedAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, diopiConstTensorHandle_t attenMask, diopiSize_t actualSeqLengths, int64_t numHeads,
                                 int64_t numKeyValueHeads, int64_t dim, diopiConstTensorHandle_t blockTable, int64_t blockSize) {
    AscendTensor qAt(q), kAt(k), vAt(v), outAt(out), blockTableAt(blockTable), attenMaskAt(attenMask);
    ASCEND_CHECK_ABORT(actualSeqLengths.len == qAt.shape(0), "The size of the first dimension of q must be equal to the length of actualSeqLengths!");
    ASCEND_CHECK_ABORT(actualSeqLengths.len == outAt.shape(0), "The size of the first dimension of out must be equal to the length of actualSeqLengths!");
    if (qAt.dim() == 2) {
        qAt = qAt.view({qAt.shape(0), (int64_t)1, qAt.shape(1)});
        outAt = outAt.view({outAt.shape(0), (int64_t)1, outAt.shape(1)});
        kAt = kAt.view({kAt.shape(0), (int64_t)1, kAt.shape(1)});
        vAt = vAt.view({vAt.shape(0), (int64_t)1, vAt.shape(1)});
    }
    if (qAt.dim() == 3) {
        ASCEND_CHECK_ABORT(1 == qAt.shape(1), "The size of the second dimension of q must be 1!");
        ASCEND_CHECK_ABORT(1 == outAt.shape(1), "The size of the second dimension of out must be 1!");
    }
    double scaleValue = 1 / std::sqrt(dim);
    std::vector<AscendTensor> keyTensors{kAt};
    std::vector<AscendTensor> valueTensors{vAt};
    int64_t innerPrecise = 1;
    AscendTensor paddingMask, dequantScale1, quantScale1, dequantScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, kvPaddingSize;
    DIOPI_ASCEND_CALL_ACLNN(aclnnIncreFlashAttentionV4,
                            ctx,
                            qAt,
                            keyTensors,
                            valueTensors,
                            paddingMask,
                            attenMaskAt,
                            actualSeqLengths,
                            dequantScale1,
                            quantScale1,
                            dequantScale2,
                            quantScale2,
                            quantOffset2,
                            antiquantScale,
                            antiquantOffset,
                            blockTableAt,
                            kvPaddingSize,
                            numHeads,
                            scaleValue,
                            "BSH",
                            numKeyValueHeads,
                            blockSize,
                            innerPrecise,
                            outAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
