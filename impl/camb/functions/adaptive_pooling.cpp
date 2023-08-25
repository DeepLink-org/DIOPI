/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);

    /* Some basic check */
    DIOPI_CHECK(inputTr.dim() == 3 || inputTr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&inputTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outputTmpTr = outputTr;
    if (inputTr.dtype() != outputTr.dtype()) {
        outputTmpTr = requiresTensor(ctx, outputTr.shape(), inputTr.dtype());
    }

    auto memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));

    DiopiTensor outputChannelLast = outputTmpTr;
    if (!outputChannelLast.isContiguous(memoryFormat)) {
        // for some special case like shape = [2, 2048, 1, 1], it's already been ChannelsLast
        outputChannelLast = requiresTensor(ctx, outputTmpTr.shape(), outputTmpTr.dtype(), diopiMemoryFormat_t::ChannelsLast);
    }

    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc outputDesc(outputChannelLast, layout);

    cnnlPoolingMode_t mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

// version should be greater than 1.15.2
#if (CNNL_MAJOR * 10000 + CNNL_MINOR * 100 + CNNL_PATCHLEVEL >= 11502)
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAdaptivePoolingForwardWorkspaceSize(handle, inputDesc.get(), mode, outputDesc.get(), &workspaceSize));

    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingForward_v2(
        handle, inputDesc.get(), inputTr.data(), mode, workspacePtr, workspaceSize, outputDesc.get(), outputChannelLast.data(), nullptr, nullptr));
#else
    DIOPI_CALLCNNL(cnnlAdaptivePoolingForward(handle, inputDesc.get(), inputTr.data(), mode, outputDesc.get(), outputChannelLast.data(), nullptr, nullptr));
#endif
    // NHWC -> NCHW
    DIOPI_CALL(impl::camb::contiguous(ctx, outputChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, outputChannelLast.tensorHandle(), outputTmpTr.tensorHandle()));

    if (outputTmpTr.dtype() != outputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTr, outputTmpTr));
    }

    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTr(input);
    DiopiTensor gradOutputTr(gradOutput);
    DiopiTensor gradInputTr(gradInput);

    /* Some basic check */
    DIOPI_CHECK(inputTr.dim() == 3 || inputTr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&gradOutputTr, &inputTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    auto memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    DIOPI_CALL(contiguous(ctx, gradOutputTr, memoryFormat));

    DiopiTensor gradInputTmpTr = gradInputTr;
    if (gradInputTr.dtype() != gradOutputTr.dtype()) {
        gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradOutputTr.dtype());
    }

    DiopiTensor gradInputChannelLast = requiresTensor(ctx, gradInputTmpTr.shape(), gradInputTmpTr.dtype(), diopiMemoryFormat_t::ChannelsLast);

    /* generate tensor desc */
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc gradOutputDesc(gradOutputTr, layout);
    CnnlTensorDesc gradInputDesc(gradInputChannelLast, layout);

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingBackward(handle,
                                               gradOutputDesc.get(),
                                               gradOutputTr.data(),
                                               nullptr,
                                               nullptr,
                                               CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                               gradInputDesc.get(),
                                               gradInputChannelLast.data()));

    // NHWC -> NCHW
    DIOPI_CALL(impl::camb::contiguous(ctx, gradInputChannelLast, diopiMemoryFormat_t::Contiguous));
    DIOPI_CALL(impl::camb::diopiCopyInp(ctx, gradInputChannelLast.tensorHandle(), gradInputTmpTr.tensorHandle()));

    if (gradInputTmpTr.dtype() != gradInputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTr, gradInputTmpTr));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
