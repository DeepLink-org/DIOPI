/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
                          diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    AscendTensor inputTensor(input);
    AscendTensor invRmsTensor(invRms);
    ASCEND_CHECK_ABORT(1 == normalizedShape.len && normalizedShape.data[0] == inputTensor.shape()[inputTensor.dim() - 1],
                       "normalized shape currently must be the size of the last dimension on ascend!");
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_bfloat16) {
        ASCEND_CHECK_ABORT(invRmsTensor.dtype() == diopi_dtype_float32, "When the dtype of input is float16 or bfloat16, the dtype of invRms must be float32!");
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnRmsNorm, ctx, input, weight, eps, out, invRms);

    if (bias) {
        diopiScalar_t alpha = constructDiopiScalarT(inputTensor.dtype(), 1.0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAdd, ctx, out, bias, &alpha);
    }
    return diopiSuccess;
}

diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                  diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRms, diopiSize_t normalizedShape, double eps) {
    AscendTensor inputTensor(input);
    AscendTensor invRmsTensor(invRms);
    ASCEND_CHECK_ABORT(1 == normalizedShape.len && normalizedShape.data[0] == inputTensor.shape()[inputTensor.dim() - 1],
                       "normalized shape currently must be the size of the last dimension on ascend!");
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_bfloat16) {
        ASCEND_CHECK_ABORT(invRmsTensor.dtype() == diopi_dtype_float32, "When the dtype of input is float16 or bfloat16, the dtype of invRms must be float32!");
    }

    AscendTensor gradWeightTensor(gradWeight);
    if (gradWeightTensor.dtype() != diopi_dtype_float32) {
        AscendTensor gradWeightTmp;
        makeTensorLike(ctx, gradWeightTmp, gradWeightTensor, diopi_dtype_float32);
        DIOPI_ASCEND_CALL_ACLNN(aclnnRmsNormGrad, ctx, gradOutput, input, invRms, weight, gradInput, gradWeightTmp);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, gradWeight, gradWeightTmp);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnRmsNormGrad, ctx, gradOutput, input, invRms, weight, gradInput, gradWeight);
    }

    if (gradBias) {
        AscendTensor gradBiasTensor(gradBias);
        AscendTensor gradOutputTensor(gradOutput);
        int64_t outDim = gradOutputTensor.dim();
        int64_t biasDim = gradBiasTensor.dim();
        if (outDim > biasDim) {
            std::vector<int64_t> sumDims(outDim - biasDim);
            std::iota(sumDims.begin(), sumDims.end(), 0);
            DIOPI_ASCEND_CALL_ACLNN(aclnnReduceSum, ctx, gradOutput, sumDims, false, gradBiasTensor.dtype(), gradBias);
        } else {
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, gradBias, gradOutput);
        }
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
