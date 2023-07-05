/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                               diopiReduction_t reduction, double beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DiopiTensor targetTensor(target);
    DiopiTensor outputTensorTemp = outputTensor;

    std::vector<DiopiTensor*> pTensors{&inputTensor, &targetTensor, &outputTensorTemp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    cnnlSmoothL1LossAlgorithm_t reductionMode;
    switch (reduction) {
        case ReductionNone:
            reductionMode = CNNL_SMOOTHL1LOSS_REDUCTION_NONE;
            break;
        case ReductionMean:
            reductionMode = CNNL_SMOOTHL1LOSS_REDUCTION_MEAN;
            break;
        case ReductionSum:
            reductionMode = CNNL_SMOOTHL1LOSS_REDUCTION_SUM;
            break;
        default:
            DIOPI_CHECK(false, "the reductionMode not supported!!")
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc(targetTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetSmoothL1LossForwardWorkspaceSize(handle, inputDesc.get(), reductionMode, &workspaceSize));

    void* workspace;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlSmoothL1LossForward_v2(handle,
                                              inputDesc.get(),
                                              inputTensor.data(),
                                              targetDesc.get(),
                                              targetTensor.data(),
                                              static_cast<float>(beta),
                                              reductionMode,
                                              workspace,
                                              workspaceSize,
                                              outputDesc.get(),
                                              outputTensorTemp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputTensorTemp));
    return diopiSuccess;
}

diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor gradInputTensor(grad_input);
    DiopiTensor gradOutputTensor(grad_output);
    DiopiTensor targetTensor(target);

    DiopiTensor gradInputTensorTemp = gradInputTensor;

    std::vector<DiopiTensor*> pTensors{&inputTensor, &targetTensor, &gradInputTensorTemp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    cnnlSmoothL1LossAlgorithm_t reductionMode;
    switch (reduction) {
        case ReductionNone:
            reductionMode = CNNL_SMOOTHL1LOSS_REDUCTION_NONE;
            break;
        case ReductionMean:
            reductionMode = CNNL_SMOOTHL1LOSS_REDUCTION_MEAN;
            break;
        case ReductionSum:
            reductionMode = CNNL_SMOOTHL1LOSS_REDUCTION_SUM;
            break;
        default:
            DIOPI_CHECK(false, "the reductionMode not supported!!")
    }

    CnnlTensorDesc gradInputDesc(gradInputTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc targetDesc(targetTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetSmoothL1LossBackwardWorkspaceSize(handle, inputDesc.get(), reductionMode, &workspaceSize));

    void* workspace;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlSmoothL1LossBackward_v2(handle,
                                               inputDesc.get(),
                                               inputTensor.data(),
                                               targetDesc.get(),
                                               targetTensor.data(),
                                               gradOutputDesc.get(),
                                               gradOutputTensor.data(),
                                               static_cast<float>(beta),
                                               reductionMode,
                                               workspace,
                                               workspaceSize,
                                               gradInputDesc.get(),
                                               gradInputTensorTemp.data()));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTemp));
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
