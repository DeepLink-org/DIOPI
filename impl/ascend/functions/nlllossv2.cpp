/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiNLLLossV2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t totalWeight, diopiConstTensorHandle_t input,
                            diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    if (input == nullptr) {
        return diopiSuccess;
    }

    AscendTensor inputAt(input);
    if (inputAt.numel() <= 0) {
        if (diopiReduction_t::ReductionMean == reduction) {
            diopiScalar_t nans{diopi_dtype_float64, std::nanf("")};
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, out, &nans);
        } else if (diopiReduction_t::ReductionSum == reduction || diopiReduction_t::ReductionNone == reduction) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, out);
        }
        return diopiSuccess;
    }

    diopiTensorHandle_t weightTmp = const_cast<diopiTensorHandle_t>(weight);
    if (weightTmp == nullptr) {
        const int64_t channel = inputAt.dim() >= 4 ? inputAt.shape(1) : inputAt.shape(-1);
        std::vector<int64_t> weightSize{channel};
        diopiSize_t weightShape = vectorToDiopiSize(weightSize);
        diopiRequireTensor(ctx, &weightTmp, &weightShape, nullptr, inputAt.dtype(), diopi_device);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, weightTmp);
    }

    if (inputAt.dim() <= 2) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnNLLLoss, ctx, input, target, weightTmp, reduction, ignoreIndex, out, totalWeight);
    } else if (inputAt.dim() == 4) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnNLLLoss2d, ctx, input, target, weightTmp, reduction, ignoreIndex, out, totalWeight);
    } else {
        AscendTensor outAt(out);
        AscendTensor targetAt(target);
        AscendTensor inputView = inputAt.view({inputAt.shape(0), inputAt.shape(1), inputAt.numel() / inputAt.shape(0) / inputAt.shape(1), 1});
        AscendTensor outView = (outAt.numel() > 1) ? outAt.view({outAt.shape(0), outAt.numel() / outAt.shape(0), 1}) : outAt;
        AscendTensor targetView = targetAt.view({targetAt.shape(0), targetAt.numel() / targetAt.shape(0), 1});
        DIOPI_ASCEND_CALL_ACLNN(aclnnNLLLoss2d, ctx, inputView, targetView, weightTmp, reduction, ignoreIndex, outView, totalWeight);
    }

    return diopiSuccess;
}

diopiError_t diopiNLLLossV2Backward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t totalWeight, diopiReduction_t reduction, int64_t ignoreIndex) {
    AscendTensor inputAt(input);
    AscendTensor gradInputAt(gradInput);
    if (input == nullptr || gradInput == nullptr || inputAt.numel() <= 0 || gradInputAt.numel() <= 0) {
        return diopiSuccess;
    }
    /*
     * A tensor representing the sum of weights for each element considered in the NLL loss computation.
     * In case a weight tensor is provided, total_weight represents the sum of weights for all the non-ignored indices in the target tensor.
     * When no weight tensor is provided, total_weight corresponds to the count of all non-ignored indices.
     */
    diopiTensorHandle_t weightTmp = const_cast<diopiTensorHandle_t>(weight);
    if (weightTmp == nullptr) {
        const int64_t channel = inputAt.dim() >= 4 ? inputAt.shape(1) : inputAt.shape(-1);
        std::vector<int64_t> weightSize{channel};
        diopiSize_t weightShape = vectorToDiopiSize(weightSize);
        diopiRequireTensor(ctx, &weightTmp, &weightShape, nullptr, inputAt.dtype(), diopi_device);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, weightTmp);
    }

    if (inputAt.dim() <= 2) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnNLLLossBackward, ctx, gradOutput, input, target, weightTmp, reduction, ignoreIndex, totalWeight, gradInput);
    } else if (inputAt.dim() == 4) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnNLLLoss2dBackward, ctx, gradOutput, input, target, weightTmp, reduction, ignoreIndex, totalWeight, gradInput);
    } else {
        AscendTensor gradIputAt(gradInput);
        AscendTensor gradOutputAt(gradOutput);
        AscendTensor targetAt(target);

        AscendTensor inputView = inputAt.view({inputAt.shape(0), inputAt.shape(1), inputAt.numel() / inputAt.shape(0) / inputAt.shape(1), 1});
        AscendTensor gradInputView =
            gradInputAt.view({gradInputAt.shape(0), gradInputAt.shape(1), gradInputAt.numel() / gradInputAt.shape(0) / gradInputAt.shape(1), 1});
        AscendTensor gradOutputView;
        if (gradOutputAt.numel() > 1) {
            gradOutputView = gradOutputAt.view({gradOutputAt.shape(0), gradOutputAt.numel() / gradOutputAt.shape(0), 1});
        } else {
            gradOutputView = gradOutputAt;
        }
        AscendTensor targetView = targetAt.view({targetAt.shape(0), targetAt.numel() / targetAt.shape(0), 1});
        DIOPI_ASCEND_CALL_ACLNN(
            aclnnNLLLoss2dBackward, ctx, gradOutputView, inputView, targetView, weightTmp, reduction, ignoreIndex, totalWeight, gradInputView);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
