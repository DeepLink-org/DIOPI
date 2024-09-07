/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include <vector>

#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                               diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t pTokenIds, diopiConstTensorHandle_t pTokenCounts,
                               diopiConstTensorHandle_t pCumsumSeqLen, int pMaxLenInBatch) {
    AscendTensor logitsAt(logits);
    AscendTensor pCumsumSeqLenAt(pCumsumSeqLen);
    AscendTensor frequencyPenaltyAt(frequencyPenalty);
    AscendTensor presencePenaltyAt(presencePenalty);
    AscendTensor pTokenIdsAt(pTokenIds);
    AscendTensor pTokenCountsAt(pTokenCounts);

    int batch = logitsAt.shape(0);
    const int64_t dim = 0;
    diopiDtype_t logitsDtype = logitsAt.dtype();

    AscendTensor curBatchIndexHostAt = deviceToHostSync(ctx, pCumsumSeqLenAt);
    AscendTensor frequencyPenaltyHostAt = deviceToHostSync(ctx, frequencyPenaltyAt);
    AscendTensor presencePenaltyHostAt = deviceToHostSync(ctx, presencePenaltyAt);

    const int* curBatchIndexData = reinterpret_cast<const int*>(curBatchIndexHostAt.data());

    for (int i = 0; i < batch; ++i) {
        int curBatchStartIndex = *(curBatchIndexData + i);
        int curBatchEndIndex = *(curBatchIndexData + (i + 1));
        AscendTensor sliceAt;
        std::vector<int64_t> sliceAtShape(1, curBatchEndIndex - curBatchStartIndex);
        makeTensor(ctx, sliceAt, sliceAtShape, diopi_dtype_int32);
        const diopiScalar_t curBatchStartIndexScalar = constructDiopiScalarT(diopi_dtype_int32, curBatchStartIndex);
        const diopiScalar_t curBatchEndIndexScalar = constructDiopiScalarT(diopi_dtype_int32, curBatchEndIndex);
        const diopiScalar_t stepScalar = constructDiopiScalarT(diopi_dtype_int32, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &curBatchStartIndexScalar, &curBatchEndIndexScalar, &stepScalar, sliceAt);

        diopiTensorHandle_t curTokenIds;
        diopiConstTensorHandle_t sliceTensorHandle = sliceAt.tensorHandle();
        ascend_npu::diopiIndex(ctx, &curTokenIds, pTokenIds, &sliceTensorHandle, 1);

        diopiTensorHandle_t curTokenCounts;
        ascend_npu::diopiIndex(ctx, &curTokenCounts, pTokenCounts, &sliceTensorHandle, 1);

        AscendTensor curTokenIdsAt(curTokenIds);
        AscendTensor curTokenCountsAt(curTokenCounts);
        AscendTensor curLogitsAt;
        std::vector<int64_t> curLogitsAtShape(1);
        curLogitsAtShape[dim] = curTokenIdsAt.shape()[0];
        makeTensor(ctx, curLogitsAt, curLogitsAtShape, logitsDtype);
        AscendTensor logitsAtI;
        makeTensor(ctx, logitsAtI, {1, logitsAt.shape()[1]}, logitsDtype);
        diopiScalar_t iScalar = constructDiopiScalarT(diopi_dtype_int32, i);
        AscendTensor iTensorAt;
        makeTensorFromScalar(ctx, iTensorAt, &iScalar, logitsAt.device());
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, logitsAt, dim, iTensorAt, logitsAtI);

        logitsAtI.view({logitsAt.shape()[1]});

        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, logitsAtI, dim, curTokenIds, curLogitsAt);
        AscendTensor frequencyPenaltyAdjustmentAt;
        makeTensor(ctx, frequencyPenaltyAdjustmentAt, curTokenCountsAt.shape(), logitsDtype);

        diopiScalar_t frequencyPenaltyAtIScalar;
        if (logitsDtype == diopi_dtype_float32) {
            const float* frequencyPenaltyData = reinterpret_cast<const float*>(frequencyPenaltyHostAt.data());
            frequencyPenaltyAtIScalar = constructDiopiScalarT(logitsDtype, *(frequencyPenaltyData + i));
        } else {
            const half_float::half* frequencyPenaltyData = reinterpret_cast<const half_float::half*>(frequencyPenaltyHostAt.data());
            frequencyPenaltyAtIScalar = constructDiopiScalarT(logitsDtype, *(frequencyPenaltyData + i));
        }
        DIOPI_ASCEND_CALL_ACLNN(aclnnMuls, ctx, curTokenCounts, &frequencyPenaltyAtIScalar, frequencyPenaltyAdjustmentAt);

        AscendTensor totalPenaltyAdjustmentAt;
        makeTensor(ctx, totalPenaltyAdjustmentAt, curTokenCountsAt.shape(), logitsDtype);

        diopiScalar_t presencePenaltyAtIScalar;
        if (logitsDtype == diopi_dtype_float32) {
            const float* presencePenaltyData = reinterpret_cast<const float*>(presencePenaltyHostAt.data());
            presencePenaltyAtIScalar = constructDiopiScalarT(logitsDtype, *(presencePenaltyData + i));
        } else {
            const half_float::half* presencePenaltyData = reinterpret_cast<const half_float::half*>(presencePenaltyHostAt.data());
            presencePenaltyAtIScalar = constructDiopiScalarT(logitsDtype, *(presencePenaltyData + i));
        }
        diopiScalar_t oneScalar = constructDiopiScalarT(logitsDtype, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnAdds, ctx, frequencyPenaltyAdjustmentAt, &presencePenaltyAtIScalar, &oneScalar, totalPenaltyAdjustmentAt);

        DIOPI_ASCEND_CALL_ACLNN(aclnnSub, ctx, curLogitsAt, totalPenaltyAdjustmentAt, &oneScalar, curLogitsAt);
        std::vector<AscendTensor> indices;
        indices.emplace_back(iTensorAt);
        indices.emplace_back(curTokenIdsAt);
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, logitsAt, indices, curLogitsAt, false, true);
    }

    return diopiSuccess;
}

diopiError_t diopiApplyPenaltyV2(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                                 diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t repetitionPenalty, diopiConstTensorHandle_t pTokenIds,
                                 diopiConstTensorHandle_t pTokenCounts) {
    AscendTensor logitsAt(logits);
    AscendTensor pTokenIdsAt(pTokenIds);
    AscendTensor pTokenCountsAt(pTokenCounts);
    AscendTensor frequencyPenaltyAt(frequencyPenalty);
    AscendTensor presencePenaltyAt(presencePenalty);

    const int32_t dim = 0;

    logitsAt.view({logitsAt.numel()});

    AscendTensor curLogitsAt;
    diopiDtype_t logitsDtype = logitsAt.dtype();
    std::vector<int64_t> pTokenIdsShape = pTokenIdsAt.shape();
    std::vector<int64_t> curLogitsAtShape = logitsAt.shape();
    curLogitsAtShape[dim] = pTokenIdsShape[0];
    makeTensor(ctx, curLogitsAt, curLogitsAtShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, logitsAt, dim, pTokenIds, curLogitsAt);

    AscendTensor repoLogitsTensorAt;
    makeTensor(ctx, repoLogitsTensorAt, curLogitsAtShape, logitsDtype);
    diopiScalar_t zeroScalar = constructDiopiScalarT(diopi_dtype_float32, 0);
    AscendTensor candTensorAt;
    makeTensor(ctx, candTensorAt, curLogitsAtShape, logitsDtype);

    DIOPI_ASCEND_CALL_ACLNN(aclnnGtScalar, ctx, curLogitsAt, &zeroScalar, candTensorAt);

    AscendTensor selfTensorAt;
    makeTensor(ctx, selfTensorAt, curLogitsAtShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnDiv, ctx, curLogitsAt, repetitionPenalty, selfTensorAt);

    AscendTensor otherTensorAt;
    makeTensor(ctx, otherTensorAt, curLogitsAtShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, curLogitsAt, repetitionPenalty, otherTensorAt);

    DIOPI_ASCEND_CALL_ACLNN(aclnnSWhere, ctx, candTensorAt, selfTensorAt, otherTensorAt, repoLogitsTensorAt);

    AscendTensor frequencyPenaltyProductAt;
    std::vector<int64_t> frequencyPenaltyProductAtShape = inferSize(pTokenCountsAt.shape(), frequencyPenaltyAt.shape());
    makeTensor(ctx, frequencyPenaltyProductAt, frequencyPenaltyProductAtShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, pTokenCounts, frequencyPenalty, frequencyPenaltyProductAt);

    AscendTensor penaltySumAt;
    std::vector<int64_t> penaltySumAtShape = inferSize(frequencyPenaltyProductAtShape, presencePenaltyAt.shape());
    makeTensor(ctx, penaltySumAt, penaltySumAtShape, logitsDtype);
    diopiScalar_t oneScalar = constructDiopiScalarT(diopi_dtype_float32, 1);
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdd, ctx, frequencyPenaltyProductAt, presencePenalty, &oneScalar, penaltySumAt);

    DIOPI_ASCEND_CALL_ACLNN(aclnnSub, ctx, repoLogitsTensorAt, penaltySumAt, &oneScalar, repoLogitsTensorAt);

    std::vector<int64_t> shape(pTokenIdsShape);
    shape.emplace_back(1);

    pTokenIdsAt.view(shape);
    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterNd, ctx, logitsAt, pTokenIdsAt, repoLogitsTensorAt, logitsAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
