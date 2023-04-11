/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
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

    auto memory_format = MemoryFormat::ChannelsLast;
    auto input_channel_last = input_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, input_tr, input_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto output_channel_last = output_tr.contiguous(ctx, memory_format);

    /* generate tensor desc */
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc input_desc(input_channel_last, layout);
    CnnlTensorDesc output_desc(output_channel_last, layout);

    /* call adaptive pooling */
    DIOPI_CALLCNNL(cnnlAdaptivePoolingForward(handle,
                                              input_desc.get(),
                                              input_channel_last.data(),
                                              CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                              output_desc.get(),
                                              output_channel_last.data(),
                                              nullptr,
                                              nullptr));

    // NHWC -> NCHW
    cnnl_transpose(ctx, handle, output_channel_last, output_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tr(input);
    DiopiTensor grad_output_tr(grad_output);
    DiopiTensor grad_input_tr(grad_input);

    /* Some basic check */
    DIOPI_CHECK(input_tr.dim() == 3 || input_tr.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    auto memory_format = MemoryFormat::ChannelsLast;
    auto grad_output_channel_last = grad_output_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, grad_output_tr, grad_output_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto grad_input_channel_last = grad_input_tr.contiguous(ctx, memory_format);

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
    cnnl_transpose(ctx, handle, grad_input_channel_last, grad_input_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
