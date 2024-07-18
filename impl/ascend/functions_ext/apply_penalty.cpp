/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include <vector>

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

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
