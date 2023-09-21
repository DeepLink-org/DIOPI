/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

/**
 * @brief Measures the Binary Cross Entropy between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param input Tensor of arbitrary shape as unnormalized scores (often referred to as logits). type = [float32, float64].
 * @param target Tensor of the same shape as input with values between 0 and 1. type = [float32, float64].
 * @param weight a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch. type = [float32, float64].
 * @param reduction Specifies the reduction to apply to the output
 * @param[out] out the output tensor. type = [float32, float64].
 */

static diopiError_t convertBCEReduction(cnnlBceLossReduction_t *bceReduction, const diopiReduction_t reduction) {
    switch (reduction) {
        case ReductionNone:
            *bceReduction = CNNL_BCE_LOSS_NONE;
            break;
        case ReductionMean:
            *bceReduction = CNNL_BCE_LOSS_MEAN;
            break;
        case ReductionSum:
            *bceReduction = CNNL_BCE_LOSS_SUM;
            break;
        default:
            DIOPI_CHECK(false, "[diopiBCELoss] unexpected bce_loss reduciton mode");
            return diopiErrorOccurred;
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor targetTensor(target);
    DiopiTensor weightTensor(weight);
    DiopiTensor outTensor(out);

    if (!weightTensor.defined()) {
        weightTensor = ones(ctx, inputTensor.shape(), inputTensor.dtype());
    }
    DIOPI_CALL(broadcastHelper(ctx, weightTensor, inputTensor, &weightTensor));

    std::vector<DiopiTensor *> tensorsVecPtr{&inputTensor, &targetTensor, &weightTensor, &outTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensorsVecPtr, supportedDtypes));
    inputTensor = *tensorsVecPtr[0];
    targetTensor = *tensorsVecPtr[1];
    weightTensor = *tensorsVecPtr[2];
    DiopiTensor outCastedTensor = *tensorsVecPtr[3];

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc(targetTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outCastedDesc(outCastedTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    void *workspace = nullptr;
    DIOPI_CALL_CNNL(cnnlGetBceLossWorkspaceSize(handle, inputDesc.get(), weightDesc.get(), &workspaceSize));
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    cnnlBceLossReduction_t bceReduction;
    convertBCEReduction(&bceReduction, reduction);

    DIOPI_CALL_CNNL(cnnlBceLoss(handle,
                                inputDesc.get(),
                                inputTensor.data(),
                                targetDesc.get(),
                                targetTensor.data(),
                                weightDesc.get(),
                                weightTensor.data(),
                                bceReduction,
                                workspace,
                                workspaceSize,
                                outCastedDesc.get(),
                                outCastedTensor.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outCastedTensor));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor targetTensor(target);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);

    if (!weightTensor.defined()) {
        weightTensor = ones(ctx, targetTensor.shape(), targetTensor.dtype());
    }
    DIOPI_CALL(broadcastHelper(ctx, weightTensor, inputTensor, &weightTensor));

    std::vector<DiopiTensor *> tensorsVecPtr{&inputTensor, &targetTensor, &weightTensor, &gradInputTensor, &gradOutputTensor};
    std::set<diopiDtype_t> supportedDtype{diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, tensorsVecPtr, supportedDtype));
    inputTensor = *tensorsVecPtr[0];
    targetTensor = *tensorsVecPtr[1];
    weightTensor = *tensorsVecPtr[2];
    DiopiTensor gradInputCastedTensor = *tensorsVecPtr[3];
    gradOutputTensor = *tensorsVecPtr[4];

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc(targetTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputCastedTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    void *workspace = nullptr;
    DIOPI_CALL_CNNL(cnnlGetBceLossBackwardWorkspaceSize(handle, targetDesc.get(), weightDesc.get(), &workspaceSize));
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    cnnlBceLossReduction_t bceReduction;
    convertBCEReduction(&bceReduction, reduction);

    DIOPI_CALL_CNNL(cnnlBceLossBackward(handle,
                                        gradOutputDesc.get(),
                                        gradOutputTensor.data(),
                                        inputDesc.get(),
                                        inputTensor.data(),
                                        targetDesc.get(),
                                        targetTensor.data(),
                                        weightDesc.get(),
                                        weightTensor.data(),
                                        bceReduction,
                                        workspace,
                                        workspaceSize,
                                        gradInputDesc.get(),
                                        gradInputCastedTensor.data()));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputCastedTensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
