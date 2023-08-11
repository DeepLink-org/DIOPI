#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalizedShape,
                            double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor saveMeanTensor(saveMean);
    DiopiTensor saveInvstdTensor(saveInvstd);

    diopiDtype_t outDtype = outTensor.dtype();
    if (outDtype != diopi_dtype_float32 && outDtype != diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, saveMeanTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, saveInvstdTensor, diopi_dtype_float32));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc saveMeanDesc(saveMeanTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize(0);
    DIOPI_CALLCNNL(cnnlGetLayerNormOpWorkspaceSize(handle, normalizedShape.len, inputDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    void *weightPtr = nullptr;
    void *biasPtr = nullptr;
    CnnlTensorDesc weightBiasDesc;
    cnnlTensorDescriptor_t weightBiasDescTmp = nullptr;
    if (weight != nullptr && bias != nullptr) {
        DiopiTensor weightTensor(weight);
        DiopiTensor biasTensor(bias);
        if (outDtype != diopi_dtype_float32 && outDtype != diopi_dtype_float16) {
            DIOPI_CALL(dataTypeCast(ctx, weightTensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, biasTensor, diopi_dtype_float32));
        }
        weightPtr = weightTensor.data();
        biasPtr = biasTensor.data();
        weightBiasDesc.set(weightTensor, CNNL_LAYOUT_ARRAY);
        weightBiasDescTmp = weightBiasDesc.get();
    }

    int axis = inputTensor.dim() - normalizedShape.len;
    DIOPI_CALLCNNL(cnnlLayerNormForward(handle,
                                        inputDesc.get(),
                                        inputTensor.data(),
                                        axis,
                                        weightBiasDescTmp,
                                        weightPtr,
                                        biasPtr,
                                        eps,
                                        workspace,
                                        workspaceSize,
                                        outDesc.get(),
                                        outTensor.data(),
                                        saveMeanDesc.get(),
                                        saveMeanTensor.data(),
                                        saveInvstdTensor.data()));

    if (outDtype != diopi_dtype_float32 && outDtype != diopi_dtype_float16) {
        DiopiTensor outTensorTmp(out);
        DiopiTensor saveMeanTensorTmp(saveMean);
        DiopiTensor saveInvstdTensorTmp(saveInvstd);
        DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, outTensor));
        DIOPI_CALL(dataTypeCast(ctx, saveMeanTensorTmp, saveMeanTensor));
        DIOPI_CALL(dataTypeCast(ctx, saveInvstdTensorTmp, saveInvstdTensor));
    }

    return diopiSuccess;
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalizedShape) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);
    DiopiTensor meanTensor(mean);
    DiopiTensor rstdTensor(rstd);
    DiopiTensor weightTensor(weight);
    DiopiTensor biasTensor(bias);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradBiasTensor(gradBias);

    diopiDtype_t outDtype = gradInputTensor.dtype();
    if (outDtype != diopi_dtype_float16 && outDtype != diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, gradOutputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, meanTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, rstdTensor, diopi_dtype_float32));
    }

    CnnlTensorDesc gradInputDesc(gradInputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc meanDesc(meanTensor, CNNL_LAYOUT_ARRAY);

    void *weightPtr = nullptr;
    CnnlTensorDesc weightBiasDesc;
    cnnlTensorDescriptor_t weightBiasDescTmp = nullptr;
    void *gradWeightPtr = nullptr;
    void *gradBiasPtr = nullptr;
    if (weight != nullptr && bias != nullptr) {
        if (outDtype != diopi_dtype_float16 && outDtype != diopi_dtype_float32) {
            DIOPI_CALL(dataTypeCast(ctx, weightTensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, gradBiasTensor, diopi_dtype_float32));
        }

        weightPtr = weightTensor.data();
        gradWeightPtr = gradWeightTensor.data();
        gradBiasPtr = gradBiasTensor.data();
        weightBiasDesc.set(weightTensor, CNNL_LAYOUT_ARRAY);
        weightBiasDescTmp = weightBiasDesc.get();
    } else {
        weightTensor = requiresTensor(ctx, normalizedShape, inputTensor.dtype());
        gradWeightTensor = requiresTensor(ctx, normalizedShape, inputTensor.dtype());
        gradBiasTensor = requiresTensor(ctx, normalizedShape, inputTensor.dtype());
        diopiScalar_t one = constructDiopiScalarT(diopi_dtype_float32, 1);
        diopiScalar_t zero = constructDiopiScalarT(diopi_dtype_float32, 0);
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(weightTensor), &one));
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(gradWeightTensor), &zero));
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(gradBiasTensor), &zero));
        weightPtr = weightTensor.data();
        weightBiasDesc.set(weightTensor, CNNL_LAYOUT_ARRAY);
        weightBiasDescTmp = weightBiasDesc.get();
        gradWeightPtr = gradWeightTensor.data();
        gradBiasPtr = gradBiasTensor.data();
    }

    int axis = inputTensor.dim() - normalizedShape.len;

    size_t workspaceSize(0);
    DIOPI_CALLCNNL(cnnlGetLayerNormBackwardWorkspaceSize(handle, inputDesc.get(), axis, &workspaceSize));
    void *workspace;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlLayerNormBackward_v2(handle,
                                            inputDesc.get(),
                                            inputTensor.data(),
                                            axis,
                                            gradOutputDesc.get(),
                                            gradOutputTensor.data(),
                                            weightBiasDescTmp,
                                            weightPtr,
                                            meanDesc.get(),
                                            meanTensor.data(),
                                            rstdTensor.data(),
                                            workspace,
                                            workspaceSize,
                                            gradInputDesc.get(),
                                            gradInputTensor.data(),
                                            gradWeightPtr,
                                            gradBiasPtr));
    if (outDtype != diopi_dtype_float16 && outDtype != diopi_dtype_float32) {
        DiopiTensor gradInputTensorTmp(gradInput);
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensorTmp, gradInputTensor));
        if (gradBias != nullptr && gradWeight != nullptr) {
            DiopiTensor gradWeightTensorTmp(gradWeight);
            DiopiTensor gradBiasTensorTmp(gradBias);
            DIOPI_CALL(dataTypeCast(ctx, gradWeightTensorTmp, gradWeightTensor));
            DIOPI_CALL(dataTypeCast(ctx, gradBiasTensorTmp, gradBiasTensor));
        }
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
