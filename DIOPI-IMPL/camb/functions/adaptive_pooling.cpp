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

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tr(input);
    DiopiTensor output_tr(out);

    /* Some basic check */
    DIOPI_CHECK(input_tr.dim() == 3 || input_tr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> p_tensors{&input_tr};
    std::set<diopiDtype_t> supported_dtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, p_tensors, supported_dtypes));

    DiopiTensor output_tmp_tr = output_tr;
    if (input_tr.dtype() != output_tr.dtype()) {
        output_tmp_tr = requiresTensor(ctx, output_tr.shape(), input_tr.dtype());
    }

    auto memory_format = MemoryFormat::ChannelsLast;
    auto input_channel_last = input_tr.contiguous(ctx, memory_format);
    DIOPI_CALL(cnnl_transpose(ctx, handle, input_tr, input_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));

    auto output_channel_last = output_tmp_tr;
    if (!output_channel_last.is_contiguous(memory_format)) {
        // for some special case like shape = [2, 2048, 1, 1], it's already been ChannelsLast
        output_channel_last = requiresTensor(ctx, output_tmp_tr.shape(), output_tmp_tr.dtype(), MemoryFormat::ChannelsLast);
    }

    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc input_desc(input_channel_last, layout);
    CnnlTensorDesc output_desc(output_channel_last, layout);

    cnnlPoolingMode_t mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    size_t workspace_size = 0;

    #if (CNNL_MAJOR >= 1 && CNNL_MINOR >= 15 && CNNL_PATCHLEVEL >= 2)

    DIOPI_CALLCNNL(cnnlGetAdaptivePoolingForwardWorkspaceSize(handle, input_desc.get(), mode, output_desc.get(), &workspace_size));

    void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingForward_v2(handle,
                                                 input_desc.get(),
                                                 input_channel_last.data(),
                                                 mode,
                                                 workspace_ptr,
                                                 workspace_size,
                                                 output_desc.get(),
                                                 output_channel_last.data(),
                                                 nullptr,
                                                 nullptr));
    #else
        DIOPI_CHECK(false, "cnnl version must be greater than 1.15.2")
    #endif

    // NHWC -> NCHW
    DIOPI_CALL(cnnl_transpose(ctx, handle, output_channel_last, output_tmp_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW));

    if (output_tmp_tr.dtype() != output_tr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output_tr, output_tmp_tr));
    }

    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tr(input);
    DiopiTensor grad_output_tr(grad_output);
    DiopiTensor grad_input_tr(grad_input);

    /* Some basic check */
    DIOPI_CHECK(input_tr.dim() == 3 || input_tr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> p_tensors{&grad_output_tr, &input_tr};
    std::set<diopiDtype_t> supported_dtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, p_tensors, supported_dtypes));

    auto memory_format = MemoryFormat::ChannelsLast;
    auto grad_output_channel_last = grad_output_tr.contiguous(ctx, memory_format);
    DIOPI_CALL(cnnl_transpose(ctx, handle, grad_output_tr, grad_output_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC));

    DiopiTensor grad_input_tmp_tr = grad_input_tr;
    if (grad_input_tr.dtype() != grad_output_tr.dtype()) {
        grad_input_tmp_tr = requiresTensor(ctx, grad_input_tr.shape(), grad_output_tr.dtype());
    }

    auto grad_input_channel_last = grad_input_tmp_tr.contiguous(ctx, memory_format);

    /* generate tensor desc */
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc grad_output_desc(grad_output_channel_last, layout);
    CnnlTensorDesc grad_input_desc(grad_input_channel_last, layout);

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingBackward(handle,
                                               grad_output_desc.get(),
                                               grad_output_channel_last.data(),
                                               nullptr,
                                               nullptr,
                                               CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                               grad_input_desc.get(),
                                               grad_input_channel_last.data()));

    // NHWC -> NCHW
    DIOPI_CALL(cnnl_transpose(ctx, handle, grad_input_channel_last, grad_input_tmp_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW));

    if (grad_input_tmp_tr.dtype() != grad_input_tr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tr, grad_input_tmp_tr));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
