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
    int dim = inputTr.dim();
    cnnlTensorLayout_t layout;
    diopiMemoryFormat_t memoryFormat;

    /* Some basic check */
    // dim == 3, without batch mode will be support in future version
    DIOPI_CHECK(dim == 4 || dim == 5, "non-empty 4D or 5D tensor expected for input");
    std::vector<DiopiTensor*> pTensors{&inputTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    if (dim == 4) {
        layout = CNNL_LAYOUT_NHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    } else if (dim == 5) {
        layout = CNNL_LAYOUT_NDHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast3d;
    }

    DiopiTensor outputTmpTr = outputTr;
    if (inputTr.dtype() != outputTr.dtype()) {
        outputTmpTr = requiresTensor(ctx, outputTr.shape(), inputTr.dtype(), memoryFormat);
    }

    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc outputDesc(outputTmpTr, layout);

    cnnlPoolingMode_t mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

// version should be greater than 1.15.2
#if (CNNL_MAJOR * 10000 + CNNL_MINOR * 100 + CNNL_PATCHLEVEL >= 11502)
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetAdaptivePoolingForwardWorkspaceSize(handle, inputDesc.get(), mode, outputDesc.get(), &workspaceSize));

    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    /* call adaptive pooling */
    DIOPI_CALL_CNNL(cnnlAdaptivePoolingForward_v2(
        handle, inputDesc.get(), inputTr.data(), mode, workspacePtr, workspaceSize, outputDesc.get(), outputTmpTr.data(), nullptr, nullptr));
#else
    DIOPI_CALL_CNNL(cnnlAdaptivePoolingForward(handle, inputDesc.get(), inputTr.data(), mode, outputDesc.get(), outputTmpTr.data(), nullptr, nullptr));
#endif

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
    int dim = inputTr.dim();
    cnnlTensorLayout_t layout;
    diopiMemoryFormat_t memoryFormat;

    /* Some basic check */
    DIOPI_CHECK(dim == 4 || dim == 5, "non-empty 4D or 5D tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&gradOutputTr, &inputTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    if (dim == 4) {
        layout = CNNL_LAYOUT_NHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    } else if (dim == 5) {
        layout = CNNL_LAYOUT_NDHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast3d;
    }

    DiopiTensor gradInputTmpTr = gradInputTr;
    if (gradInputTr.dtype() != gradOutputTr.dtype()) {
        gradInputTmpTr = requiresTensor(ctx, gradInputTr.shape(), gradOutputTr.dtype(), memoryFormat);
    }

    /* generate tensor desc */
    CnnlTensorDesc gradOutputDesc(gradOutputTr, layout);
    CnnlTensorDesc gradInputDesc(gradInputTmpTr, layout);

    /* call adaptive pooling */
    DIOPI_CALL_CNNL(cnnlAdaptivePoolingBackward(handle,
                                                gradOutputDesc.get(),
                                                gradOutputTr.data(),
                                                nullptr,
                                                nullptr,
                                                CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                                gradInputDesc.get(),
                                                gradInputTmpTr.data()));

    if (gradInputTmpTr.dtype() != gradInputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTr, gradInputTmpTr));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
