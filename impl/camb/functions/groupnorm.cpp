/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "cmath"
namespace impl {
namespace camb {

DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t numGroups,
                                      double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor biasTensor(bias);
    DiopiTensor outTensor(out);
    DiopiTensor saveMeanTensor(saveMean);
    DiopiTensor saveInvstdTensor(saveInvstd);

    if (!weightTensor.defined()) {
        weightTensor = ones(ctx, {outTensor.shape()[1]}, inputTensor.dtype());
    }
    if (!biasTensor.defined()) {
        biasTensor = zeros(ctx, {outTensor.shape()[1]}, inputTensor.dtype());
    }

    if (inputTensor.defined() && inputTensor.numel() == 0) {
        return diopiSuccess;
    }
    int inputDim = inputTensor.dim();
    if (inputDim == 2) {
        inputTensor.unsqueeze(2);
        inputTensor.unsqueeze(3);
        outTensor.view(inputTensor.shape());
    } else if (inputDim == 3) {
        inputTensor.unsqueeze(3);
        outTensor.view(inputTensor.shape());
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc saveMeanDesc(saveMeanTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetGroupNormForwardWorkspaceSize(handle, numGroups, inputDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    if (eps < 0) {
        eps = 1e-5;
    }
    DIOPI_CALLCNNL(cnnlGroupNormForward_v3(handle,
                                           eps,
                                           numGroups,
                                           inputDesc.get(),
                                           inputTensor.data(),
                                           weightDesc.get(),
                                           weightTensor.data(),
                                           biasTensor.data(),
                                           workspace,
                                           workspaceSize,
                                           outDesc.get(),
                                           outTensor.data(),
                                           saveMeanDesc.get(),
                                           saveMeanTensor.data(),
                                           saveInvstdTensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                              diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd,
                                              int64_t numGroups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor meanTensor(mean);
    DiopiTensor rstdTensor(rstd);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradBiasTensor(gradBias);
    DiopiTensor gradOutputTensor(gradOutput);

    if (inputTensor.defined() && inputTensor.numel() == 0) {
        if (inputTensor.shape()[0] == 0) {
            diopiScalar_t zero = constructDiopiScalarT(inputTensor.dtype(), 0);
            DIOPI_CALL(diopiFill(ctx, gradWeight, &zero));
            DIOPI_CALL(diopiFill(ctx, gradBias, &zero));
            return diopiSuccess;
        } else {
            diopiScalar_t nan = constructDiopiScalarT(inputTensor.dtype(), NAN);
            diopiScalar_t zero = constructDiopiScalarT(inputTensor.dtype(), 0);
            DIOPI_CALL(diopiFill(ctx, gradWeight, &nan));
            DIOPI_CALL(diopiFill(ctx, gradBias, &zero));
            return diopiSuccess;
        }
    }

    int inputDim = inputTensor.dim();
    if (inputDim == 2) {
        inputTensor.unsqueeze(2);
        inputTensor.unsqueeze(3);
        gradInputTensor.view(inputTensor.shape());
        gradOutputTensor.view(inputTensor.shape());
    } else if (inputDim == 3) {
        inputTensor.unsqueeze(3);
        gradInputTensor.view(inputTensor.shape());
        gradOutputTensor.view(inputTensor.shape());
    }

    if (!weightTensor.defined()) {
        weightTensor = ones(ctx, {gradOutputTensor.shape()[1]}, inputTensor.dtype());
    }
    if (!gradWeightTensor.defined()) {
        gradWeightTensor = ones(ctx, {gradOutputTensor.shape()[1]}, inputTensor.dtype());
    }
    if (!gradBiasTensor.defined()) {
        gradBiasTensor = ones(ctx, {gradOutputTensor.shape()[1]}, inputTensor.dtype());
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc meanDesc(meanTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc rstdDesc(rstdTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputTensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc gradWeightDesc(gradWeightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradBiasDesc(gradBiasTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_NCHW);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetGroupNormBackwardWorkspaceSize(handle, inputTensor.shape()[0] * inputTensor.shape()[1], &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlGroupNormBackward(handle,
                                         inputDesc.get(),
                                         inputTensor.data(),
                                         gradOutputDesc.get(),
                                         gradOutputTensor.data(),
                                         weightDesc.get(),
                                         weightTensor.data(),
                                         meanDesc.get(),
                                         meanTensor.data(),
                                         rstdDesc.get(),
                                         rstdTensor.data(),
                                         numGroups,
                                         gradInputDesc.get(),
                                         gradInputTensor.data(),
                                         gradWeightDesc.get(),
                                         gradWeightTensor.data(),
                                         gradBiasDesc.get(),
                                         gradBiasTensor.data(),
                                         workspace,
                                         workspaceSize));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
