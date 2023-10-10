/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions_mmcv.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"
#include "../diopi_helper.hpp"

extern "C" DIOPI_API diopiError_t diopiModulatedDeformConvMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t columns,
                                                               diopiTensorHandle_t ones, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                                               diopiConstTensorHandle_t bias, diopiConstTensorHandle_t offset, diopiConstTensorHandle_t mask,
                                                               int64_t kernelH, int64_t kernelW, const int64_t strideH, const int64_t strideW,
                                                               const int64_t padH, const int64_t padW, const int64_t dilationH, const int64_t dilationW,
                                                               const int64_t group, const int64_t deformableGroup, const bool withBias) {
    cnnlHandle_t handle = impl::camb::cnnlHandlePool.get(ctx);

    impl::camb::DiopiTensor outputTensor(output);
    impl::camb::DiopiTensor inputTensor(input);
    impl::camb::DiopiTensor weightTensor(weight);
    impl::camb::DiopiTensor offsetTensor(offset);
    impl::camb::DiopiTensor maskTensor(mask);
    impl::camb::DiopiTensor biasTensor(bias);

    impl::camb::printDevData(ctx, outputTensor, "outputTensor", 5);
    impl::camb::printDevData(ctx, inputTensor, "inputTensor", 5);
    impl::camb::printDevData(ctx, offsetTensor, "offsetTensor", 5);
    impl::camb::printDevData(ctx, maskTensor, "maskTensor", 5);
    std::cout << "strideH: " << strideH << std::endl;
    std::cout << "strideW: " << strideW << std::endl;
    std::cout << "padH: " << padH << std::endl;
    std::cout << "padW: " << padW << std::endl;
    std::cout << "kernelH: " << kernelH << std::endl;
    std::cout << "kernelW: " << kernelW << std::endl;
    std::cout << "group: " << group << std::endl;
    std::cout << "deformableGroup: " << deformableGroup << std::endl;

    // check input sizes
    DIOPI_CHECK(inputTensor.dim() == 4, "input's dim must equal to 4");
    DIOPI_CHECK(weightTensor.dim() == 4, "weight's dim must equal to 4");
    DIOPI_CHECK(offsetTensor.dim() == 4, "offset's dim must equal to 4");
    DIOPI_CHECK(maskTensor.dim() == 4, "mask's dim must equal to 4");

    std::vector<impl::camb::DiopiTensor *> tensors{&inputTensor, &weightTensor, &offsetTensor, &maskTensor, &biasTensor};
    DIOPI_CALL(impl::camb::autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    impl::camb::DiopiTensor outputTensorTmp = outputTensor;
    if (inputTensor.dtype() != outputTensor.dtype()) {
        outputTensorTmp = impl::camb::requiresTensor(ctx, outputTensor.shape(), inputTensor.dtype());
    }

    auto memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    DIOPI_CALL(impl::camb::contiguous(ctx, inputTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, weightTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, offsetTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, maskTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, biasTensor, diopiMemoryFormat_t::Contiguous));

    auto outputChannelLast = impl::camb::requiresTensor(ctx, outputTensorTmp.shape(), outputTensorTmp.dtype(), memoryFormat);

    impl::camb::CnnlTensorDesc outputDesc(outputChannelLast, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc offsetDesc(offsetTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc maskDesc(maskTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc biasDesc(biasTensor, CNNL_LAYOUT_ARRAY);

    int32_t batchSize = static_cast<int32_t>(inputTensor.size(0));
    // im2col_step should be set in cnnl kernel.
    int32_t im2colStep = batchSize;
    int32_t stride[2] = {static_cast<int32_t>(strideH), static_cast<int32_t>(strideW)};
    int32_t padding[4] = {static_cast<int32_t>(padH), static_cast<int32_t>(padH), static_cast<int32_t>(padW), static_cast<int32_t>(padW)};
    int32_t dilation[2] = {static_cast<int32_t>(dilationH), static_cast<int32_t>(dilationW)};

    cnnlDCNDescriptor_t dcnDesc;
    DIOPI_CALL_CNNL(cnnlCreateDCNDescriptor(&dcnDesc));
    DIOPI_CALL_CNNL(cnnlSetDCNDescriptor(dcnDesc,
                                         inputTensor.dim(),
                                         padding,
                                         stride,
                                         dilation,
                                         static_cast<int32_t>(deformableGroup),
                                         static_cast<int32_t>(group),
                                         im2colStep,
                                         CNNL_DTYPE_FLOAT));

    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetDCNForwardWorkspaceSize(handle,
                                                   dcnDesc,
                                                   inputDesc.get(),
                                                   offsetDesc.get(),
                                                   maskDesc.get(),
                                                   weightDesc.get(),
                                                   withBias ? biasDesc.get() : nullptr,
                                                   outputDesc.get(),
                                                   &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = impl::camb::requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALL_CNNL(cnnlDCNForward(handle,
                                   dcnDesc,
                                   inputDesc.get(),
                                   inputTensor.data(),
                                   offsetDesc.get(),
                                   offsetTensor.data(),
                                   maskDesc.get(),
                                   maskTensor.data(),
                                   weightDesc.get(),
                                   weightTensor.data(),
                                   withBias ? biasDesc.get() : nullptr,
                                   withBias ? biasTensor.data() : nullptr,
                                   workspace,
                                   workspaceSize,
                                   outputDesc.get(),
                                   outputChannelLast.data()));
    DIOPI_CALL_CNNL(cnnlDestroyDCNDescriptor(dcnDesc));

    // NHWC -> NCHW
    DIOPI_CALL(impl::camb::contiguous(ctx, outputChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, outputChannelLast.tensorHandle(), outputTensorTmp.tensorHandle()));
    if (outputTensor.dtype() != outputTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, outputTensor, outputTensorTmp));
    }

    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiModulatedDeformConvBackwardMmcv(
    diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias, diopiTensorHandle_t gradOffset,
    diopiTensorHandle_t gradMask, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t ones,
    diopiConstTensorHandle_t offset, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t columns, diopiConstTensorHandle_t gradOutput, int64_t kernelH,
    int64_t kernelW, int64_t strideH, int64_t strideW, int64_t padH, int64_t padW, int64_t dilationH, int64_t dilationW, int64_t group, int64_t deformableGroup,
    const bool withBias) {
    cnnlHandle_t handle = impl::camb::cnnlHandlePool.get(ctx);

    impl::camb::DiopiTensor gradInputTensor(gradInput);
    impl::camb::DiopiTensor gradWeightTensor(gradWeight);
    impl::camb::DiopiTensor gradOffsetTensor(gradOffset);
    impl::camb::DiopiTensor gradMaskTensor(gradMask);
    impl::camb::DiopiTensor gradBiasTensor(gradBias);

    impl::camb::DiopiTensor inputTensor(input);
    impl::camb::DiopiTensor weightTensor(weight);
    impl::camb::DiopiTensor offsetTensor(offset);
    impl::camb::DiopiTensor maskTensor(mask);
    impl::camb::DiopiTensor biasTensor(bias);
    impl::camb::DiopiTensor gradOutputTensor(gradOutput);

    impl::camb::printDevData(ctx, gradOutputTensor, "gradOutputTensor", 5);
    impl::camb::printDevData(ctx, gradMaskTensor, "gradMaskTensor", 5);
    impl::camb::printDevData(ctx, gradOffsetTensor, "gradOffsetTensor", 5);
    impl::camb::printDevData(ctx, maskTensor, "maskTensor", 5);
    impl::camb::printDevData(ctx, offsetTensor, "offsetTensor", 5);

    std::vector<impl::camb::DiopiTensor *> tensors{&inputTensor, &weightTensor, &offsetTensor, &maskTensor, &biasTensor, &gradOutputTensor};
    DIOPI_CALL(impl::camb::autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    impl::camb::DiopiTensor gradInputTensorTmp = gradInputTensor;
    impl::camb::DiopiTensor gradWeightTensorTmp = gradWeightTensor;
    impl::camb::DiopiTensor gradOffsetTensorTmp = gradOffsetTensor;
    impl::camb::DiopiTensor gradMaskTensorTmp = gradMaskTensor;
    impl::camb::DiopiTensor gradBiasTensorTmp = gradBiasTensor;
    if (inputTensor.dtype() != gradInputTensor.dtype()) {
        gradInputTensorTmp = impl::camb::requiresTensor(ctx, gradInputTensor.shape(), inputTensor.dtype());
    }
    if (inputTensor.dtype() != gradWeightTensor.dtype()) {
        gradWeightTensorTmp = impl::camb::requiresTensor(ctx, gradWeightTensor.shape(), inputTensor.dtype());
    }
    if (inputTensor.dtype() != gradOffsetTensor.dtype()) {
        gradOffsetTensorTmp = impl::camb::requiresTensor(ctx, gradOffsetTensor.shape(), inputTensor.dtype());
    }
    if (inputTensor.dtype() != gradMaskTensor.dtype()) {
        gradMaskTensorTmp = impl::camb::requiresTensor(ctx, gradMaskTensor.shape(), inputTensor.dtype());
    }
    if (inputTensor.dtype() != gradBiasTensor.dtype()) {
        gradBiasTensorTmp = impl::camb::requiresTensor(ctx, gradBiasTensor.shape(), inputTensor.dtype());
    }

    auto memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    DIOPI_CALL(impl::camb::contiguous(ctx, inputTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, weightTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, offsetTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, maskTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, gradOutputTensor, memoryFormat));
    DIOPI_CALL(impl::camb::contiguous(ctx, biasTensor, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::contiguous(ctx, gradBiasTensorTmp, diopiMemoryFormat_t::Contiguous));

    auto gradInputChannelLast = impl::camb::requiresTensor(ctx, gradInputTensorTmp.shape(), gradInputTensorTmp.dtype(), memoryFormat);
    auto gradWeightChannelLast = impl::camb::requiresTensor(ctx, gradWeightTensorTmp.shape(), gradWeightTensorTmp.dtype(), memoryFormat);
    auto gradOffsetChannelLast = impl::camb::requiresTensor(ctx, gradOffsetTensorTmp.shape(), gradOffsetTensorTmp.dtype(), memoryFormat);
    auto gradMaskChannelLast = impl::camb::requiresTensor(ctx, gradMaskTensorTmp.shape(), gradMaskTensorTmp.dtype(), memoryFormat);

    impl::camb::CnnlTensorDesc gradInputDesc(gradInputChannelLast, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc gradWeightDesc(gradWeightChannelLast, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc gradOffsetDesc(gradOffsetChannelLast, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc gradMaskDesc(gradMaskChannelLast, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc gradBiasDesc(gradBiasTensorTmp, CNNL_LAYOUT_ARRAY);

    impl::camb::CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc offsetDesc(offsetTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc maskDesc(maskTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_NHWC);
    impl::camb::CnnlTensorDesc biasDesc(biasTensor, CNNL_LAYOUT_ARRAY);

    int32_t batchSize = static_cast<int32_t>(inputTensor.size(0));
    // im2col_step should be set in cnnl kernel.
    int32_t im2colStep = batchSize;
    int32_t stride[2] = {static_cast<int32_t>(strideH), static_cast<int32_t>(strideW)};
    int32_t padding[4] = {static_cast<int32_t>(padH), static_cast<int32_t>(padH), static_cast<int32_t>(padW), static_cast<int32_t>(padW)};
    int32_t dilation[2] = {static_cast<int32_t>(dilationH), static_cast<int32_t>(dilationW)};

    cnnlDCNDescriptor_t dcnDesc;
    DIOPI_CALL_CNNL(cnnlCreateDCNDescriptor(&dcnDesc));
    DIOPI_CALL_CNNL(cnnlSetDCNDescriptor(dcnDesc,
                                         inputTensor.dim(),
                                         padding,
                                         stride,
                                         dilation,
                                         static_cast<int32_t>(deformableGroup),
                                         static_cast<int32_t>(group),
                                         im2colStep,
                                         CNNL_DTYPE_FLOAT));

    size_t dataWorkspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetDCNBakcwardDataWorkspaceSize(handle,
                                                        dcnDesc,
                                                        inputDesc.get(),
                                                        offsetDesc.get(),
                                                        maskDesc.get(),
                                                        weightDesc.get(),
                                                        gradOutputDesc.get(),
                                                        gradInputDesc.get(),
                                                        gradOffsetDesc.get(),
                                                        gradMaskDesc.get(),
                                                        &dataWorkspaceSize));
    void *dataWorkspace = nullptr;
    if (dataWorkspaceSize != 0) {
        dataWorkspace = impl::camb::requiresBuffer(ctx, dataWorkspaceSize).data();
    }

    // grad_input, grad_offset, grad_mask
    DIOPI_CALL_CNNL(cnnlDCNBackwardData(handle,
                                        dcnDesc,
                                        inputDesc.get(),
                                        inputTensor.data(),
                                        offsetDesc.get(),
                                        offsetTensor.data(),
                                        maskDesc.get(),
                                        maskTensor.data(),
                                        weightDesc.get(),
                                        weightTensor.data(),
                                        gradOutputDesc.get(),
                                        gradOutputTensor.data(),
                                        dataWorkspace,
                                        dataWorkspaceSize,
                                        gradInputDesc.get(),
                                        gradInputChannelLast.data(),
                                        gradOffsetDesc.get(),
                                        gradOffsetChannelLast.data(),
                                        gradMaskDesc.get(),
                                        gradMaskChannelLast.data()));

    size_t weightWorkspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetDCNBackwardWeightWorkspaceSize(handle,
                                                          dcnDesc,
                                                          inputDesc.get(),
                                                          offsetDesc.get(),
                                                          maskDesc.get(),
                                                          gradOutputDesc.get(),
                                                          gradWeightDesc.get(),
                                                          withBias ? gradBiasDesc.get() : nullptr,
                                                          &weightWorkspaceSize));
    void *weightWorkspace = nullptr;
    if (weightWorkspaceSize != 0) {
        weightWorkspace = impl::camb::requiresBuffer(ctx, weightWorkspaceSize).data();
    }

    // grad_bias, grad_weight
    DIOPI_CALL_CNNL(cnnlDCNBackwardWeight(handle,
                                          dcnDesc,
                                          inputDesc.get(),
                                          inputTensor.data(),
                                          offsetDesc.get(),
                                          offsetTensor.data(),
                                          maskDesc.get(),
                                          maskTensor.data(),
                                          gradOutputDesc.get(),
                                          gradOutputTensor.data(),
                                          weightWorkspace,
                                          weightWorkspaceSize,
                                          gradWeightDesc.get(),
                                          gradWeightChannelLast.data(),
                                          withBias ? gradBiasDesc.get() : nullptr,
                                          withBias ? gradBiasTensorTmp.data() : nullptr));

    DIOPI_CALL_CNNL(cnnlDestroyDCNDescriptor(dcnDesc));

    // NHWC -> NCHW
    DIOPI_CALL(impl::camb::contiguous(ctx, gradInputChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradInputChannelLast.tensorHandle(), gradInputTensorTmp.tensorHandle()));
    if (gradInputTensor.dtype() != gradInputTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));
    }

    DIOPI_CALL(impl::camb::contiguous(ctx, gradOffsetChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradOffsetChannelLast.tensorHandle(), gradOffsetTensorTmp.tensorHandle()));
    if (gradOffsetTensor.dtype() != gradOffsetTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradOffsetTensor, gradOffsetTensorTmp));
    }

    DIOPI_CALL(impl::camb::contiguous(ctx, gradMaskChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradMaskChannelLast.tensorHandle(), gradMaskTensorTmp.tensorHandle()));
    if (gradMaskTensor.dtype() != gradMaskTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradMaskTensor, gradMaskTensorTmp));
    }

    DIOPI_CALL(impl::camb::contiguous(ctx, gradWeightChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradWeightChannelLast.tensorHandle(), gradWeightTensorTmp.tensorHandle()));
    if (gradWeightTensor.dtype() != gradWeightTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradWeightTensor, gradWeightTensorTmp));
    }

    if (gradBiasTensor.dtype() != gradBiasTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradBiasTensor, gradBiasTensorTmp));
    }

    return diopiSuccess;
}