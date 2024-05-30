/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions_mmcv.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
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

    // check input sizes
    DIOPI_CHECK(inputTensor.dim() == 4, "input's dim must equal to 4");
    DIOPI_CHECK(weightTensor.dim() == 4, "weight's dim must equal to 4");
    DIOPI_CHECK(offsetTensor.dim() == 4, "offset's dim must equal to 4");
    DIOPI_CHECK(maskTensor.dim() == 4, "mask's dim must equal to 4");

    // check for cnnl kernel
    DIOPI_CHECK(outputTensor.size(2) == offsetTensor.size(2) && outputTensor.size(2) == maskTensor.size(2) && outputTensor.size(3) == offsetTensor.size(3) &&
                    outputTensor.size(3) == maskTensor.size(3),
                "offset and mask should have the same spatial size as the output of the convolution");

    std::vector<impl::camb::DiopiTensor*> tensors{&inputTensor, &weightTensor, &offsetTensor, &maskTensor, &biasTensor};
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
    impl::camb::DiopiTensor strideTensor = impl::camb::requiresTensor(ctx, {2}, diopi_dtype_int32, diopiDevice_t::diopi_host);
    impl::camb::DiopiTensor paddingTensor = impl::camb::requiresTensor(ctx, {4}, diopi_dtype_int32, diopiDevice_t::diopi_host);
    impl::camb::DiopiTensor dilationTensor = impl::camb::requiresTensor(ctx, {2}, diopi_dtype_int32, diopiDevice_t::diopi_host);
    int32_t* strideTensorPtr = (int32_t*)strideTensor.data();
    int32_t* paddingTensorPtr = (int32_t*)paddingTensor.data();
    int32_t* dilationTensorPtr = (int32_t*)dilationTensor.data();
    strideTensorPtr[0] = static_cast<int32_t>(strideH);
    strideTensorPtr[1] = static_cast<int32_t>(strideW);
    paddingTensorPtr[0] = static_cast<int32_t>(padH);
    paddingTensorPtr[1] = static_cast<int32_t>(padH);
    paddingTensorPtr[2] = static_cast<int32_t>(padW);
    paddingTensorPtr[3] = static_cast<int32_t>(padW);
    dilationTensorPtr[0] = static_cast<int32_t>(dilationH);
    dilationTensorPtr[1] = static_cast<int32_t>(dilationW);

    cnnlDCNDescriptor_t dcnDesc;
    DIOPI_CALL_CNNL(cnnlCreateDCNDescriptor(&dcnDesc));
    DIOPI_CALL_CNNL(cnnlSetDCNDescriptor(dcnDesc,
                                         inputTensor.dim(),
                                         paddingTensorPtr,
                                         strideTensorPtr,
                                         dilationTensorPtr,
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
    void* workspace = nullptr;
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
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, outputChannelLast.tensorHandle(), outputTensorTmp.tensorHandle()));
    if (outputTensor.dtype() != outputTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, outputTensor, outputTensorTmp));
    }

    // TODO: replace cnrtQueueSync with asynchronous code
    // padding,stride,dilation is host tensor, may be released ealier
    cnrtQueueSync(getStream(ctx));

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

    // check input sizes
    DIOPI_CHECK(inputTensor.dim() == 4, "input's dim must equal to 4");
    DIOPI_CHECK(weightTensor.dim() == 4, "weight's dim must equal to 4");
    DIOPI_CHECK(offsetTensor.dim() == 4, "offset's dim must equal to 4");
    DIOPI_CHECK(maskTensor.dim() == 4, "mask's dim must equal to 4");

    // check for cnnl kernel
    DIOPI_CHECK(gradOutputTensor.size(2) == offsetTensor.size(2) && gradOutputTensor.size(2) == maskTensor.size(2) &&
                    gradOutputTensor.size(3) == offsetTensor.size(3) && gradOutputTensor.size(3) == maskTensor.size(3),
                "offset and mask should have the same spatial size as the output of the convolution");

    std::vector<impl::camb::DiopiTensor*> tensors{&inputTensor, &weightTensor, &offsetTensor, &maskTensor, &biasTensor, &gradOutputTensor};
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
    impl::camb::DiopiTensor strideTensor = impl::camb::requiresTensor(ctx, {2}, diopi_dtype_int32, diopiDevice_t::diopi_host);
    impl::camb::DiopiTensor paddingTensor = impl::camb::requiresTensor(ctx, {4}, diopi_dtype_int32, diopiDevice_t::diopi_host);
    impl::camb::DiopiTensor dilationTensor = impl::camb::requiresTensor(ctx, {2}, diopi_dtype_int32, diopiDevice_t::diopi_host);
    int32_t* strideTensorPtr = (int32_t*)strideTensor.data();
    int32_t* paddingTensorPtr = (int32_t*)paddingTensor.data();
    int32_t* dilationTensorPtr = (int32_t*)dilationTensor.data();
    strideTensorPtr[0] = static_cast<int32_t>(strideH);
    strideTensorPtr[1] = static_cast<int32_t>(strideW);
    paddingTensorPtr[0] = static_cast<int32_t>(padH);
    paddingTensorPtr[1] = static_cast<int32_t>(padH);
    paddingTensorPtr[2] = static_cast<int32_t>(padW);
    paddingTensorPtr[3] = static_cast<int32_t>(padW);
    dilationTensorPtr[0] = static_cast<int32_t>(dilationH);
    dilationTensorPtr[1] = static_cast<int32_t>(dilationW);

    cnnlDCNDescriptor_t dcnDesc;
    DIOPI_CALL_CNNL(cnnlCreateDCNDescriptor(&dcnDesc));
    DIOPI_CALL_CNNL(cnnlSetDCNDescriptor(dcnDesc,
                                         inputTensor.dim(),
                                         paddingTensorPtr,
                                         strideTensorPtr,
                                         dilationTensorPtr,
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
    void* dataWorkspace = nullptr;
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
    void* weightWorkspace = nullptr;
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
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradInputChannelLast.tensorHandle(), gradInputTensorTmp.tensorHandle()));
    if (gradInputTensor.dtype() != gradInputTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));
    }

    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradOffsetChannelLast.tensorHandle(), gradOffsetTensorTmp.tensorHandle()));
    if (gradOffsetTensor.dtype() != gradOffsetTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradOffsetTensor, gradOffsetTensorTmp));
    }

    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradMaskChannelLast.tensorHandle(), gradMaskTensorTmp.tensorHandle()));
    if (gradMaskTensor.dtype() != gradMaskTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradMaskTensor, gradMaskTensorTmp));
    }

    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradWeightChannelLast.tensorHandle(), gradWeightTensorTmp.tensorHandle()));
    if (gradWeightTensor.dtype() != gradWeightTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradWeightTensor, gradWeightTensorTmp));
    }

    if (gradBiasTensor.dtype() != gradBiasTensorTmp.dtype()) {
        DIOPI_CALL(impl::camb::dataTypeCast(ctx, gradBiasTensor, gradBiasTensorTmp));
    }

    // TODO: replace cnrtQueueSync with asynchronous code
    // padding,stride,dilation is host tensor, may be released ealier
    cnrtQueueSync(getStream(ctx));

    return diopiSuccess;
}
