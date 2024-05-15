/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalizedShape,
                            double eps) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    if (0 == inputAt.numel()) {
        diopiScalar_t zeroScalar = constructDiopiScalarT(outAt.dtype(), 0.0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, out, &zeroScalar);
        return diopiSuccess;
    }

    diopiTensorHandle_t weightTemp = createTensorIfNullptrOrConstCast(ctx, weight, normalizedShape, inputAt.dtype(), true, 1);
    diopiTensorHandle_t biasTemp = createTensorIfNullptrOrConstCast(ctx, bias, normalizedShape, inputAt.dtype(), true, 0);

    // gen begin normalized dim
    diopiSize_t inShape;
    diopiGetTensorShape(input, &inShape);
    const int axis = inShape.len - normalizedShape.len;
    int64_t beginDim = axis;

    // call aclnnLayerNorm
    DIOPI_ASCEND_CALL_ACLNN(aclnnLayerNorm, ctx, input, normalizedShape, weightTemp, biasTemp, eps, out, saveMean, saveInvstd);
    return diopiSuccess;
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalizedShape) {
    AscendTensor inputAt(input);
    diopiTensorHandle_t weightTemp = createTensorIfNullptrOrConstCast(ctx, weight, normalizedShape, inputAt.dtype(), true, 1);
    diopiTensorHandle_t biasTemp = createTensorIfNullptrOrConstCast(ctx, bias, normalizedShape, inputAt.dtype(), true, 0);
    diopiTensorHandle_t gradWeightTemp = createTensorIfNullptrOrConstCast(ctx, gradWeight, normalizedShape, inputAt.dtype(), false, 0);
    diopiTensorHandle_t gradBiasTemp = createTensorIfNullptrOrConstCast(ctx, gradBias, normalizedShape, inputAt.dtype(), false, 0);

    // Align the shape of mean and rstd with input
    AscendTensor meanAt(mean), rstdAt(rstd);
    while (meanAt.dim() < inputAt.dim()) {
        meanAt.unsqueeze(meanAt.dim());
        rstdAt.unsqueeze(rstdAt.dim());
    }

    int64_t gradMaskData[3] = {true, true, true};
    if (nullptr == gradInput) {
        gradMaskData[0] = false;
    }
    if (nullptr == gradWeight) {
        gradMaskData[1] = false;
    }
    if (nullptr == gradBias) {
        gradMaskData[2] = false;
    }

    diopiSize_t gradMask{gradMaskData, 3};
    DIOPI_ASCEND_CALL_ACLNN(aclnnLayerNormBackward,
                            ctx,
                            gradOutput,
                            inputAt,
                            normalizedShape,
                            meanAt,
                            rstdAt,
                            weightTemp,
                            biasTemp,
                            gradMask,
                            gradInput,
                            gradWeightTemp,
                            gradBiasTemp);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
