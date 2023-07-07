/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t posWeight, diopiReduction_t reduction) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor targetTensor(target);
    DiopiTensor weightTensor(weight);
    DiopiTensor posWeightTensor(posWeight);
    DiopiTensor outTensor(out);

    bool weightFlag = true;
    bool posWeightFlag = true;
    if (!weight) {
        weightFlag = false;
    }
    if (!posWeight) {
        posWeightFlag = false;
    }

    std::vector<DiopiTensor*> inTensors{&inputTensor, &targetTensor};
    DIOPI_CALL(autoCastTensorType(ctx, inTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor inputTensorTmp = *inTensors[0];
    DiopiTensor targetTensorTmp = *inTensors[1];
    DiopiTensor outTensorTmp = outTensor;
    if (outTensorTmp.dtype() != inputTensorTmp.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), inputTensorTmp.dtype());
    }

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc(targetTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    DiopiTensor weightTensorTmp;
    DiopiTensor posWeightTensorTmp;
    CnnlTensorDesc weightDesc;
    CnnlTensorDesc posWeightDesc;
    if (weightFlag) {
        std::vector<DiopiTensor*> wTensors{&weightTensor};
        DIOPI_CALL(autoCastTensorType(ctx, wTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        weightTensorTmp = *wTensors[0];
        weightDesc.set(weightTensorTmp, CNNL_LAYOUT_ARRAY);
    }
    if (posWeightFlag) {
        std::vector<DiopiTensor*> poTensors{&posWeightTensor};
        DIOPI_CALL(autoCastTensorType(ctx, poTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        posWeightTensorTmp = *poTensors[0];
        posWeightDesc.set(posWeightTensorTmp, CNNL_LAYOUT_ARRAY);
    }

    cnnlBceWithLogitsReduction_t reductionMode;
    switch (reduction) {
        case 0:
            reductionMode = CNNL_BCE_WITH_LOGITS_NONE;
            break;
        case 1:
            reductionMode = CNNL_BCE_WITH_LOGITS_MEAN;
            break;
        case 2:
            reductionMode = CNNL_BCE_WITH_LOGITS_SUM;
            break;
        default:
            DIOPI_CHECK(false, "bce_with_logits reduction parameter is not avaliable");
            break;
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetBceWithLogitsWorkspaceSize(
        handle, inputDesc.get(), weightFlag ? weightDesc.get() : nullptr, posWeightFlag ? posWeightDesc.get() : nullptr, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    cnnlComputationPreference_t mode = CNNL_COMPUTATION_FAST;
    DIOPI_CALLCNNL(cnnlBceWithLogits_v2(handle,
                                        mode,
                                        inputDesc.get(),
                                        inputTensorTmp.data(),
                                        targetDesc.get(),
                                        targetTensorTmp.data(),
                                        weightFlag ? weightDesc.get() : nullptr,
                                        weightFlag ? weightTensorTmp.data() : nullptr,
                                        posWeightFlag ? posWeightDesc.get() : nullptr,
                                        posWeightFlag ? posWeightTensorTmp.data() : nullptr,
                                        reductionMode,
                                        workspace,
                                        workspaceSize,
                                        outDesc.get(),
                                        outTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t posWeight, diopiReduction_t reduction) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);
    DiopiTensor targetTensor(target);
    DiopiTensor weightTensor(weight);
    DiopiTensor posWeightTensor(posWeight);
    DiopiTensor gradInputTensor(gradInput);

    bool weightFlag = true;
    bool posWeightFlag = true;
    if (!weight) {
        weightFlag = false;
    }
    if (!posWeight) {
        posWeightFlag = false;
    }

    std::vector<DiopiTensor*> inTensors{&gradOutputTensor, &inputTensor, &targetTensor};
    DIOPI_CALL(autoCastTensorType(ctx, inTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor gradOutputTensorTmp = *inTensors[0];
    DiopiTensor inputTensorTmp = *inTensors[1];
    DiopiTensor targetTensorTmp = *inTensors[2];
    DiopiTensor gradInputTensorTmp = gradInputTensor;
    if (gradInputTensorTmp.dtype() != inputTensorTmp.dtype()) {
        gradInputTensorTmp = requiresTensor(ctx, gradInputTensor.shape(), inputTensorTmp.dtype());
    }

    CnnlTensorDesc gradOutputDesc(gradOutputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc(targetTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputTensorTmp, CNNL_LAYOUT_ARRAY);

    DiopiTensor weightTensorTmp;
    DiopiTensor posWeightTensorTmp;
    CnnlTensorDesc weightDesc;
    CnnlTensorDesc posWeightDesc;
    if (weightFlag) {
        std::vector<DiopiTensor*> wTensors{&weightTensor};
        DIOPI_CALL(autoCastTensorType(ctx, wTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        weightTensorTmp = *wTensors[0];
        weightDesc.set(weightTensorTmp, CNNL_LAYOUT_ARRAY);
    }
    if (posWeightFlag) {
        std::vector<DiopiTensor*> poTensors{&posWeightTensor};
        DIOPI_CALL(autoCastTensorType(ctx, poTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        posWeightTensorTmp = *poTensors[0];
        posWeightDesc.set(posWeightTensorTmp, CNNL_LAYOUT_ARRAY);
    }

    cnnlBceWithLogitsReduction_t reductionMode;
    switch (reduction) {
        case 0:
            reductionMode = CNNL_BCE_WITH_LOGITS_NONE;
            break;
        case 1:
            reductionMode = CNNL_BCE_WITH_LOGITS_MEAN;
            break;
        case 2:
            reductionMode = CNNL_BCE_WITH_LOGITS_SUM;
            break;
        default:
            DIOPI_CHECK(false, "bce_with_logits reduction parameter is not avaliable");
            break;
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetBceWithLogitsBackwardWorkspaceSize(
        handle, targetDesc.get(), weightFlag ? weightDesc.get() : nullptr, posWeightFlag ? posWeightDesc.get() : nullptr, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlBceWithLogitsBackward(handle,
                                             gradOutputDesc.get(),
                                             gradOutputTensorTmp.data(),
                                             inputDesc.get(),
                                             inputTensorTmp.data(),
                                             targetDesc.get(),
                                             targetTensorTmp.data(),
                                             weightFlag ? weightDesc.get() : nullptr,
                                             weightFlag ? weightTensorTmp.data() : nullptr,
                                             posWeightFlag ? posWeightDesc.get() : nullptr,
                                             posWeightFlag ? posWeightTensorTmp.data() : nullptr,
                                             reductionMode,
                                             workspace,
                                             workspaceSize,
                                             gradInputDesc.get(),
                                             gradInputTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
