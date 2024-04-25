/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions_ext.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../mlu_helper.hpp"

namespace impl {
namespace camb {

DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
                                    diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor biasTensor(bias);
    DiopiTensor outTensor(out);
    DiopiTensor invRmsTensor(invRms);
    DIOPI_CHECK((inputTensor.dim() == invRmsTensor.dim())|| (inputTensor.dim() - normalizedShape.len == invRmsTensor.dim()
    ), "dimension error in RMSNORM-invRmsTensor");
    DIOPI_CHECK(outTensor.shape() == inputTensor.shape(), "dimension error in RMSNORM");

    // zero-shape protection
    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    // get normalized axis
    int axis = inputTensor.dim() - normalizedShape.len;
    for (int i = 0; i < normalizedShape.len; i++) {
        DIOPI_CHECK(inputTensor.shape()[axis + i] == normalizedShape.data[i], "cnnl can only normalized last x dimensions");
    }

    // change input,output data type
    // mlu370 supports float16, float32
    // mlu590 supports float16, float32, bfloat16
    std::vector<DiopiTensor*> inTensors;
    if (biasTensor.defined()) {
        inTensors = {&inputTensor, &weightTensor, &biasTensor};
    } else {
        inTensors = {&inputTensor, &weightTensor};
    }
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float16};
    DIOPI_CALL(autoCastTensorType(ctx, inTensors, supportedDtypes));

    DiopiTensor outTmpTr = outTensor;
    if (outTensor.dtype() != inputTensor.dtype()) {
        outTmpTr = requiresTensor(ctx, outTensor.shape(), outTensor.stride(), inputTensor.dtype());
    }

    DiopiTensor invRmsTmpTr = invRmsTensor;
    if (invRmsTmpTr.dtype() != inputTensor.dtype()) {
        invRmsTmpTr = requiresTensor(ctx, invRmsTmpTr.shape(), invRmsTmpTr.stride(), inputTensor.dtype());
    }

    // set Tensors' decriptor
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc wbDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTmpDesc(outTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invRmsTmpDesc(invRmsTmpTr, CNNL_LAYOUT_ARRAY);

    // get worksize
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetRmsNormOpWorkspaceSize(handle, axis, inputDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();
    
    DIOPI_CALL_CNNL(cnnlRmsNormForward(handle,
                                       axis,
                                       inputDesc.get(),
                                       inputTensor.data(),
                                       wbDesc.get(),
                                       weightTensor.data(),
                                       biasTensor.defined() ? biasTensor.data() : nullptr,
                                       static_cast<float>(eps),
                                       workspace,
                                       workspaceSize,
                                       outTmpDesc.get(),
                                       outTmpTr.data(),
                                       invRmsTmpDesc.get(),
                                       invRmsTmpTr.data()));

    if (outTmpTr.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTmpTr));
    }

    if (invRmsTmpTr.dtype() != invRmsTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, invRmsTensor, invRmsTmpTr));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRms,
                                            diopiSize_t normalizedShape, double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor invRmsTensor(invRms);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradBiasTensor(gradBias);
    DIOPI_CHECK((inputTensor.dim() == invRmsTensor.dim())||(inputTensor.dim() - normalizedShape.len == invRmsTensor.dim()), "dimension error in RMSNORM-invRmsTensor");   
    DIOPI_CHECK(inputTensor.shape() == gradInputTensor.shape(), "dimension error in RMSNORM");

    // zero-shape protection
    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    // get normalized axis
    int axis = inputTensor.dim() - normalizedShape.len;
    for (int i = 0; i < normalizedShape.len; i++) {
        DIOPI_CHECK(inputTensor.shape()[axis + i] == normalizedShape.data[i], "cnnl can only normalized last x dimensions");
    }

    // change input,output data type
    // mlu370 supports float16, float32
    // mlu590 supports float16, float32, bfloat16
    std::vector<DiopiTensor*> inTensors{&inputTensor, &weightTensor, &gradOutputTensor, &invRmsTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float16};
    DIOPI_CALL(autoCastTensorType(ctx, inTensors, supportedDtypes));

    DiopiTensor gradInputTmpTr = gradInputTensor;
    if (gradInputTensor.dtype() != inputTensor.dtype()) {
        gradInputTmpTr = requiresTensor(ctx, gradInputTensor.shape(), gradInputTensor.stride(), inputTensor.dtype());
    }

    DiopiTensor gradWeightTmpTr = gradWeightTensor;
    if (gradWeightTensor.dtype() != inputTensor.dtype()) {
        gradWeightTmpTr = requiresTensor(ctx, gradWeightTensor.shape(), gradWeightTensor.stride(), inputTensor.dtype());
    }

    DiopiTensor gradBiasTmpTr = gradBiasTensor;
    if (gradBiasTensor.defined()) {
        if (gradBiasTensor.dtype() != inputTensor.dtype()) {
            gradBiasTmpTr = requiresTensor(ctx, gradBiasTensor.shape(), gradBiasTensor.stride(), inputTensor.dtype());
        }
    }

    // set Tensors' decriptor
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invRmsDesc(invRmsTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradWBDesc(gradWeightTmpTr, CNNL_LAYOUT_ARRAY);

    // get worksize
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetRmsNormBackwardWorkspaceSize(handle, inputDesc.get(), axis, &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALL_CNNL(cnnlRmsNormBackward(handle,
                                        axis,
                                        inputDesc.get(),
                                        inputTensor.data(),
                                        gradOutputDesc.get(),
                                        gradOutputTensor.data(),
                                        weightDesc.get(),
                                        weightTensor.data(),
                                        invRmsDesc.get(),
                                        invRmsTensor.data(),
                                        workspace,
                                        workspaceSize,
                                        gradInputDesc.get(),
                                        gradInputTmpTr.data(),
                                        gradWBDesc.get(),
                                        gradWeightTmpTr.data(),
                                        gradBiasTensor.defined() ? gradBiasTmpTr.data() : nullptr));

    if (gradInputTmpTr.dtype() != gradInputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTmpTr));
    }

    if (gradWeightTmpTr.dtype() != gradWeightTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightTmpTr));
    }

    if (gradBiasTensor.defined()) {
        if (gradBiasTmpTr.dtype() != gradBiasTensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, gradBiasTensor, gradBiasTmpTr));
        }
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
