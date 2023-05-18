/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t diopiTensorPermute(diopiContextHandle_t ctx, DiopiTensor &dst_tensor, DiopiTensor src_tensor, std::vector<int64_t> perm_axis) {
    if (!dst_tensor.defined()) {
        std::vector<int64_t> src_shape_t_64(src_tensor.shape().size());
        for (int i = 0; i < src_tensor.shape().size(); ++i) {
            src_shape_t_64[i] = src_tensor.shape()[perm_axis[i]];
        }
        diopiSize_t src_t_shape(src_shape_t_64.data(), src_shape_t_64.size());
        auto dst_handle = dst_tensor.tensorHandle();
        DIOPI_CALL(diopiRequireTensor(ctx, &dst_handle, &src_t_shape, nullptr, src_tensor.dtype(), diopi_device));
        dst_tensor = DiopiTensor(dst_handle);
    }
    diopiSize_t axis_size(perm_axis.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, dst_tensor.tensorHandle(), src_tensor.tensorHandle(), axis_size));
    return diopiSuccess;
}

diopiError_t diopiTensorPermute2D(diopiContextHandle_t ctx, DiopiTensor &dst, DiopiTensor src, MemoryFormat format) {
    if (src.is_contiguous(format)) {
        dst = src;
        return diopiSuccess;
    }
    if (src.is_contiguous(MemoryFormat::Contiguous) && format == MemoryFormat::ChannelsLast) {
        DIOPI_CALL(diopiTensorPermute(ctx, dst, src, {0, 2, 3, 1}));
        return diopiSuccess;
    }
    if (src.is_contiguous(MemoryFormat::ChannelsLast) && format == MemoryFormat::Contiguous) {
        DIOPI_CALL(diopiTensorPermute(ctx, dst, src, {0, 3, 1, 2}))
    }
    return diopiErrorOccurred;
}

}  // namespace

extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor weight_tensor(weight);
    DiopiTensor output_tensor(out);

    DIOPI_CHECK(true, input_tensor.is_contiguous() || input_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2d] the memory format is not supportted.");
    DIOPI_CHECK(true, weight_tensor.is_contiguous() || weight_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2d] the memory format is not supportted.");
    DIOPI_CHECK(true, output_tensor.is_contiguous() || output_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2d] the memory format is not supportted.");

    DiopiTensor input_tensor_casted = input_tensor;
    DiopiTensor weight_tensor_casted = weight_tensor;
    DiopiTensor output_tensor_casted = output_tensor;

    std::vector<DiopiTensor *> tensors{&input_tensor_casted, &weight_tensor_casted, &output_tensor_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor input_tensor_t, weight_tensor_t, output_tensor_t;

    DIOPI_CALL(diopiTensorPermute2D(ctx, input_tensor_t, input_tensor_casted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(diopiTensorPermute2D(ctx, output_tensor_t, output_tensor_casted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(diopiTensorPermute2D(ctx, weight_tensor_t, weight_tensor_casted, MemoryFormat::ChannelsLast));

    std::vector<int32_t> input_t_shape{input_tensor_t.shape().begin(), input_tensor_t.shape().end()};
    std::vector<int32_t> weight_t_shape{weight_tensor_t.shape().begin(), weight_tensor_t.shape().end()};
    std::vector<int32_t> output_t_shape{output_tensor_t.shape().begin(), output_tensor_t.shape().end()};

    CnnlTensorDesc input_desc(input_tensor_t, CNNL_LAYOUT_NHWC, input_t_shape);
    CnnlTensorDesc weight_desc(weight_tensor_t, CNNL_LAYOUT_NHWC, weight_t_shape);
    CnnlTensorDesc output_desc(output_tensor_t, CNNL_LAYOUT_NHWC, output_t_shape);

    DiopiTensor bias_tensor(bias);
    DiopiTensor bias_tensor_casted = bias_tensor;
    CnnlTensorDesc bias_desc;
    if (nullptr != bias) {
        std::vector<DiopiTensor *> tensors{&bias_tensor_casted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        DIOPI_CALL(bias_desc.set(bias_tensor_casted, CNNL_LAYOUT_ARRAY));
    }

    std::vector<int> stride_vec{stride.data, stride.data + stride.len};
    std::vector<int> padding_vec{padding.data, padding.data + padding.len};
    std::vector<int> dilation_vec{dilation.data, dilation.data + dilation.len};

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> conv_desc;

    int padding_[4] = {padding_vec[0], padding_vec[0], padding_vec[1], padding_vec[1]};
    int stride_[2] = {stride_vec[0], stride_vec[1]};
    int dilation_[2] = {dilation_vec[0], dilation_vec[1]};

    cnnlDataType_t compute_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&compute_type, input_tensor_t.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(conv_desc.get(), 4, padding_, stride_, dilation_, groups, compute_type));

    size_t workspace_size;
    DIOPI_CALLCNNL(cnnlGetConvolutionForwardWorkspaceSize(
        handle, input_desc.get(), weight_desc.get(), output_desc.get(), bias_desc.get(), conv_desc.get(), CNNL_CONVOLUTION_FWD_ALGO_DIRECT, &workspace_size));

    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionForward(handle,
                                          conv_desc.get(),
                                          CNNL_CONVOLUTION_FWD_ALGO_DIRECT,
                                          NULL,
                                          input_desc.get(),
                                          input_tensor_t.data(),
                                          weight_desc.get(),
                                          weight_tensor_t.data(),
                                          bias_tensor.defined() ? bias_desc.get() : nullptr,
                                          bias_tensor.defined() ? bias_tensor_casted.data() : nullptr,
                                          workspace,
                                          workspace_size,
                                          NULL,
                                          output_desc.get(),
                                          output_tensor_t.data()));

    DIOPI_CALL(diopiTensorPermute2D(ctx, output_tensor_casted, output_tensor_casted, MemoryFormat::Contiguous));
    DIOPI_CALL(dataTypeCast(ctx, output_tensor, output_tensor_casted));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                   diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                   diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor weight_tensor(weight);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_weight_tensor(grad_weight);

    DIOPI_CHECK(true, input_tensor.is_contiguous() || input_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(true, weight_tensor.is_contiguous() || weight_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(true, grad_output_tensor.is_contiguous() || grad_output_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(true, grad_input_tensor.is_contiguous() || grad_input_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2dBackward] the memory format is not supportted.");
    DIOPI_CHECK(true, grad_weight_tensor.is_contiguous() || grad_weight_tensor.is_contiguous(MemoryFormat::ChannelsLast),
                "[diopiConvolution2dBackward] the memory format is not supportted.");

    DiopiTensor input_casted = input_tensor;
    DiopiTensor weight_casted = weight_tensor;
    DiopiTensor grad_output_casted = grad_output_tensor;
    DiopiTensor grad_input_casted = grad_input_tensor;
    DiopiTensor grad_weight_casted = grad_weight_tensor;

    std::vector<DiopiTensor *> tensors{&input_casted, &weight_casted, &grad_output_casted, &grad_input_casted, &grad_weight_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    DiopiTensor input_t, weight_t, grad_output_t, grad_input_t, grad_weight_t;

    DIOPI_CALL(diopiTensorPermute2D(ctx, input_t, input_casted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(diopiTensorPermute2D(ctx, weight_t, weight_casted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(diopiTensorPermute2D(ctx, grad_input_t, grad_input_casted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(diopiTensorPermute2D(ctx, grad_output_t, grad_output_casted, MemoryFormat::ChannelsLast));
    DIOPI_CALL(diopiTensorPermute2D(ctx, grad_weight_t, grad_weight_casted, MemoryFormat::ChannelsLast));

    std::vector<int32_t> input_t_shape{input_t.shape().begin(), input_t.shape().end()};
    std::vector<int32_t> weight_t_shape{weight_t.shape().begin(), weight_t.shape().end()};
    std::vector<int32_t> grad_output_t_shape{grad_output_t.shape().begin(), grad_output_t.shape().end()};
    std::vector<int32_t> grad_input_t_shape{grad_input_t.shape().begin(), grad_input_t.shape().end()};
    std::vector<int32_t> grad_weight_shape{grad_weight_t.shape().begin(), grad_weight_t.shape().end()};

    CnnlTensorDesc input_desc(input_t, CNNL_LAYOUT_NHWC, input_t_shape);
    CnnlTensorDesc weight_desc(weight_t, CNNL_LAYOUT_NHWC, weight_t_shape);
    CnnlTensorDesc output_grad_desc(grad_output_t, CNNL_LAYOUT_NHWC, grad_output_t_shape);
    CnnlTensorDesc input_grad_desc(grad_input_t, CNNL_LAYOUT_NHWC, grad_input_t_shape);
    CnnlTensorDesc weight_grad_desc(grad_weight_t, CNNL_LAYOUT_NHWC, grad_weight_shape);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> conv_desc;

    std::vector<int> stride_vec{stride.data, stride.data + stride.len};
    std::vector<int> padding_vec{padding.data, padding.data + padding.len};
    std::vector<int> dilation_vec{dilation.data, dilation.data + dilation.len};

    int padding_[4] = {padding_vec[0], padding_vec[1], padding_vec[0], padding_vec[1]};
    int stride_[2] = {stride_vec[0], stride_vec[1]};
    int dilation_[2] = {dilation_vec[0], dilation_vec[1]};

    cnnlDataType_t compute_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&compute_type, input_t.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(conv_desc.get(), 4, padding_, stride_, dilation_, groups, compute_type));

    size_t workspace_size_filter = 0;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardFilterWorkspaceSize(
        handle, input_desc.get(), output_grad_desc.get(), weight_desc.get(), conv_desc.get(), CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT, &workspace_size_filter));

    void *workspace_filter = nullptr;
    if (workspace_size_filter != 0) {
        workspace_filter = requiresBuffer(ctx, workspace_size_filter).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionBackwardFilter(handle,
                                                 NULL,
                                                 input_desc.get(),
                                                 input_t.data(),
                                                 output_grad_desc.get(),
                                                 grad_output_t.data(),
                                                 conv_desc.get(),
                                                 CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT,
                                                 workspace_filter,
                                                 workspace_size_filter,
                                                 NULL,
                                                 weight_grad_desc.get(),
                                                 grad_weight_t.data()));

    size_t workspace_size_input;
    DIOPI_CALLCNNL(cnnlGetConvolutionBackwardDataWorkspaceSize(handle,
                                                               weight_desc.get(),
                                                               output_grad_desc.get(),
                                                               conv_desc.get(),
                                                               input_grad_desc.get(),
                                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                                               &workspace_size_input));

    void *workspace_input;
    if (workspace_size_input != 0) {
        workspace_input = requiresBuffer(ctx, workspace_size_input).data();
    }

    DIOPI_CALLCNNL(cnnlConvolutionBackwardData(handle,
                                               NULL,
                                               weight_desc.get(),
                                               weight_t.data(),
                                               output_grad_desc.get(),
                                               grad_output_t.data(),
                                               conv_desc.get(),
                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                               workspace_input,
                                               workspace_size_input,
                                               NULL,
                                               input_grad_desc.get(),
                                               grad_input_t.data()));

    DIOPI_CALL(diopiTensorPermute2D(ctx, grad_input_casted, grad_input_t, MemoryFormat::Contiguous));
    DIOPI_CALL(diopiTensorPermute2D(ctx, grad_weight_casted, grad_weight_t, MemoryFormat::Contiguous));

    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, grad_input_casted));
    DIOPI_CALL(dataTypeCast(ctx, grad_weight_tensor, grad_weight_casted));

    if (grad3 != nullptr) {
        DiopiTensor bias_grad_tensor(grad3);
        DiopiTensor grad_bias_casted = bias_grad_tensor;
        std::vector<DiopiTensor *> tensors{&grad_bias_casted};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
        CnnlTensorDesc bias_grad_desc(grad_bias_casted, CNNL_LAYOUT_ARRAY);
        std::vector<int64_t> bias_shape{bias_grad_tensor.shape().begin(), bias_grad_tensor.shape().end()};
        bias_sizes->data = bias_shape.data();
        bias_sizes->len = bias_shape.size();
        size_t workspace_size_bias;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, output_grad_desc.get(), bias_grad_desc.get(), 3, &workspace_size_bias))
        void *workspace_bias = nullptr;
        if (0 != workspace_size_bias) {
            workspace_bias = requiresBuffer(ctx, workspace_size_bias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(
            handle, output_grad_desc.get(), grad_output_t.data(), 3, bias_grad_desc.get(), grad_bias_casted.data(), workspace_bias, workspace_size_bias));
        DIOPI_CALL(dataTypeCast(ctx, bias_grad_tensor, grad_bias_casted))
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
