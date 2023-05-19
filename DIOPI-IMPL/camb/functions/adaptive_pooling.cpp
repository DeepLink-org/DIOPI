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

    auto memoryFormat = MemoryFormat::ChannelsLast;
    auto inputChannelLast = inputTr.contiguous(ctx, memoryFormat);
    DIOPI_CALL(cnnlTranspose(ctx, handle, inputTr, inputChannelLast, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));

    auto outputChannelLast = outputTmpTr;
    if (!outputChannelLast.isContiguous(memoryFormat)) {
        // for some special case like shape = [2, 2048, 1, 1], it's already been ChannelsLast
        outputChannelLast = requiresTensor(ctx, outputTmpTr.shape(), outputTmpTr.dtype(), MemoryFormat::ChannelsLast);
    }

    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc inputDesc(inputChannelLast, layout);
    CnnlTensorDesc outputDesc(outputChannelLast, layout);

    cnnlPoolingMode_t mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAdaptivePoolingForwardWorkspaceSize(handle, inputDesc.get(), mode, outputDesc.get(), &workspaceSize));

    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingForward_v2(handle,
                                                 inputDesc.get(),
                                                 inputChannelLast.data(),
                                                 mode,
                                                 workspacePtr,
                                                 workspaceSize,
                                                 outputDesc.get(),
                                                 outputChannelLast.data(),
                                                 nullptr,
                                                 nullptr));

    // NHWC -> NCHW
    DIOPI_CALL(cnnlTranspose(ctx, handle, outputChannelLast, outputTmpTr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW));

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

    auto memoryFormat = MemoryFormat::ChannelsLast;
    auto gradOutputChannelLast = gradOutputTr.contiguous(ctx, memoryFormat);
    DIOPI_CALL(cnnlTranspose(ctx, handle, gradOutputTr, gradOutputChannelLast, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));

    DiopiTensor gradInputTmpTr = gradInputTr;
    if (gradInputTr.dtype() != gradOutputTr.dtype()) {
        gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradOutputTr.dtype());
    }

    auto gradInputChannelLast = gradInputTmpTr.contiguous(ctx, memoryFormat);

    /* generate tensor desc */
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc gradOutputDesc(gradOutputChannelLast, layout);
    CnnlTensorDesc gradInputDesc(gradInputChannelLast, layout);

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingBackward(handle,
                                               gradOutputDesc.get(),
                                               gradOutputChannelLast.data(),
                                               nullptr,
                                               nullptr,
                                               CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                               gradInputDesc.get(),
                                               gradInputChannelLast.data()));

    // NHWC -> NCHW
    DIOPI_CALL(cnnlTranspose(ctx, handle, gradInputChannelLast, gradInputTmpTr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW));

    if (gradInputTmpTr.dtype() != gradInputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTr, gradInputTmpTr));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
