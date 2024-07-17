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

    AscendTensor curLogits;
    diopiDtype_t logitsDtype = logitsAt.dtype();
    std::vector<int64_t> pTokenIdsShape = pTokenIdsAt.shape();
    std::vector<int64_t> curLogitsShape = logitsAt.shape();
    curLogitsShape[dim] = pTokenIdsShape.data()[0];
    makeTensor(ctx, curLogits, curLogitsShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, logitsAt, dim, pTokenIds, curLogits);

    AscendTensor repoLogitsTensor;
    makeTensor(ctx, repoLogitsTensor, curLogitsShape, logitsDtype);
    diopiScalar_t zeroScalar = constructDiopiScalarT(diopi_dtype_float32, 0);
    AscendTensor candTensor;
    makeTensor(ctx, candTensor, curLogitsShape, logitsDtype);

    DIOPI_ASCEND_CALL_ACLNN(aclnnGtScalar, ctx, curLogits, &zeroScalar, candTensor);

    AscendTensor selfTensor;
    makeTensor(ctx, selfTensor, curLogitsShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnDiv, ctx, curLogits, repetitionPenalty, selfTensor);

    AscendTensor otherTensor;
    makeTensor(ctx, otherTensor, curLogitsShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, curLogits, repetitionPenalty, otherTensor);

    DIOPI_ASCEND_CALL_ACLNN(aclnnSWhere, ctx, candTensor, selfTensor, otherTensor, repoLogitsTensor);

    AscendTensor frequencyPenaltyProduct;
    std::vector<int64_t> frequencyPenaltyProductShape = inferSize(pTokenCountsAt.shape(), frequencyPenaltyAt.shape());
    makeTensor(ctx, frequencyPenaltyProduct, frequencyPenaltyProductShape, logitsDtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, pTokenCounts, frequencyPenalty, frequencyPenaltyProduct);

    AscendTensor penaltySum;
    std::vector<int64_t> penaltySumShape = inferSize(frequencyPenaltyProductShape, presencePenaltyAt.shape());
    makeTensor(ctx, penaltySum, penaltySumShape, logitsDtype);
    diopiScalar_t oneScalar = constructDiopiScalarT(diopi_dtype_float32, 1);
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdd, ctx, frequencyPenaltyProduct, presencePenalty, &oneScalar, penaltySum);

    DIOPI_ASCEND_CALL_ACLNN(aclnnSub, ctx, repoLogitsTensor, penaltySum, &oneScalar, repoLogitsTensor);

    std::vector<int64_t> shape(pTokenIdsShape);
    shape.emplace_back(1);

    pTokenIdsAt.view(shape);
    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterNd, ctx, logitsAt, pTokenIdsAt, repoLogitsTensor, logitsAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
