/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
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

    int dim = 0;

    logitsAt.view({logitsAt.numel()});

    diopiTensorHandle_t curLogits;
    diopiDtype_t logitsDtype;
    diopiGetTensorDtype(logits, &logitsDtype);
    diopiSize_t pTokenIdsSize;
    diopiGetTensorShape(pTokenIds, &pTokenIdsSize);
    std::vector<int64_t> pTokenIdsShape(pTokenIdsSize.data, pTokenIdsSize.data + pTokenIdsSize.len);
    std::vector<int64_t> curLogitsShape = logitsAt.shape();
    curLogitsShape[dim] = pTokenIdsShape.data()[0];
    diopiSize_t curLogitsSize = {curLogitsShape.data(), static_cast<int64_t>(curLogitsShape.size())};
    diopiSize_t logitsStride;
    diopiGetTensorStride(logits, &logitsStride);
    diopiRequireTensor(ctx, &curLogits, &curLogitsSize, &logitsStride, logitsDtype, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, logitsAt, dim, pTokenIds, curLogits);

    diopiTensorHandle_t repoLogitsTensor;
    diopiRequireTensor(ctx, &repoLogitsTensor, &curLogitsSize, &logitsStride, logitsDtype, diopi_device);
    diopiScalar_t zeroScalar = constructDiopiScalarT(diopi_dtype_float32, 0);
    diopiTensorHandle_t candTensor;
    diopiRequireTensor(ctx, &candTensor, &curLogitsSize, &logitsStride, logitsDtype, diopi_device);

    DIOPI_ASCEND_CALL_ACLNN(aclnnGtScalar, ctx, curLogits, &zeroScalar, candTensor);

    diopiTensorHandle_t selfTensor;
    diopiRequireTensor(ctx, &selfTensor, &curLogitsSize, &logitsStride, logitsDtype, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnDiv, ctx, curLogits, repetitionPenalty, selfTensor);

    diopiTensorHandle_t otherTensor;
    diopiRequireTensor(ctx, &otherTensor, &curLogitsSize, &logitsStride, logitsDtype, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, curLogits, repetitionPenalty, otherTensor);

    DIOPI_ASCEND_CALL_ACLNN(aclnnSWhere, ctx, candTensor, selfTensor, otherTensor, repoLogitsTensor);

    diopiTensorHandle_t frequencyPenaltyProduct;
    std::vector<int64_t> frequencyPenaltyProductShape = inferSize(pTokenCountsAt.shape(), frequencyPenaltyAt.shape());
    diopiSize_t frequencyPenaltyProductSize = {frequencyPenaltyProductShape.data(), static_cast<int64_t>(frequencyPenaltyProductShape.size())};
    diopiRequireTensor(ctx, &frequencyPenaltyProduct, &frequencyPenaltyProductSize, &logitsStride, logitsDtype, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, pTokenCounts, frequencyPenalty, frequencyPenaltyProduct);

    diopiTensorHandle_t penaltySum;
    std::vector<int64_t> penaltySumShape = inferSize(frequencyPenaltyProductShape, presencePenaltyAt.shape());
    diopiSize_t penaltySumSize = {penaltySumShape.data(), static_cast<int64_t>(penaltySumShape.size())};
    diopiRequireTensor(ctx, &penaltySum, &penaltySumSize, &logitsStride, logitsDtype, diopi_device);
    diopiScalar_t oneScalar = constructDiopiScalarT(diopi_dtype_float32, 1);
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdd, ctx, frequencyPenaltyProduct, presencePenalty, &oneScalar, penaltySum);

    DIOPI_ASCEND_CALL_ACLNN(aclnnSub, ctx, repoLogitsTensor, penaltySum, &oneScalar, repoLogitsTensor);

    std::vector<int64_t> shape(pTokenIdsAt.shape());
    shape.push_back(1);

    pTokenIdsAt.view(shape);
    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterNd, ctx, logitsAt, pTokenIdsAt, repoLogitsTensor, logitsAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
