/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {
extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t out,
                                           diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias,
                                           diopiSize_t stride,
                                           diopiSize_t padding,
                                           diopiSize_t dilation,
                                           int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);
    auto weight_tensor = DiopiTensor(weight);

    diopiTensorHandle_t input_t;
    diopiTensorHandle_t weight_t;
    diopiTensorHandle_t output_t;

    auto permute_to_nhwc = [&](auto src, auto &dst) {
        std::vector<int64_t> axis{0, 2, 3, 1};
        auto src_tensor = DiopiTensor(src);
        std::vector<int64_t> src_shape_t_64(src_tensor.shape().size());
        for (int i = 0; i < src_tensor.shape().size(); ++i) {
            src_shape_t_64[i] = src_tensor.shape()[axis[i]];
        }
        diopiSize_t src_t_shape(src_shape_t_64.data(), src_shape_t_64.size());
        DIOPI_CALL(diopiRequireTensor(ctx, &dst, &src_t_shape, nullptr, src_tensor.dtype(), diopi_device));
        diopiSize_t nchw2nhwc(axis.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, dst, src, nchw2nhwc));
        return diopiSuccess;
    };

    DIOPI_CALL(permute_to_nhwc(input, input_t));
    DIOPI_CALL(permute_to_nhwc(weight, weight_t));
    DIOPI_CALL(permute_to_nhwc(out, output_t));

    CnnlTensorDesc input_desc, output_desc, weight_desc;
    DIOPI_CALL(input_desc.set(input_tensor, CNNL_LAYOUT_NHWC));
    DIOPI_CALL(output_desc.set(output_tensor, CNNL_LAYOUT_NHWC));
    DIOPI_CALL(weight_desc.set(weight_tensor, CNNL_LAYOUT_NHWC));

    const void *bias_ptr = nullptr;
    CnnlTensorDesc bias_desc;
    if (nullptr != bias) {
        auto bias_tensor = DiopiTensor(bias);
        DIOPI_CALL(bias_desc.set(bias_tensor, CNNL_LAYOUT_ARRAY));
        bias_ptr = bias_tensor.data();
    }

    std::vector<int> stride_vec{stride.data, stride.data + stride.len};
    std::vector<int> padding_vec{padding.data, padding.data + padding.len};
    std::vector<int> dilation_vec{dilation.data, dilation.data + dilation.len};

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> conv_desc;

    int padding_[4] = {padding_vec[0], padding_vec[1], padding_vec[0], padding_vec[1]};
    int stride_[2] = {stride_vec[0], stride_vec[1]};
    int dilation_[2] = {dilation_vec[0], dilation_vec[1]};

    cnnlDataType_t input_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&input_type, input_tensor.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(conv_desc.get(), 4, padding_, stride_, dilation_, groups, input_type));

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
                                          DiopiTensor(input_t).data(),
                                          weight_desc.get(),
                                          DiopiTensor(weight_t).data(),
                                          bias_desc.get(),
                                          bias_ptr,
                                          workspace,
                                          workspace_size,
                                          NULL,
                                          output_desc.get(),
                                          DiopiTensor(output_t).data()));

    std::vector<int64_t> perm_nhwc2nchw{0, 3, 1, 2};
    diopiSize_t nhwc2nchw(perm_nhwc2nchw.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, out, output_t, nhwc2nchw));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx,
                                                   diopiTensorHandle_t grad_input,
                                                   diopiTensorHandle_t grad_weight,
                                                   diopiTensorHandle_t grad3,
                                                   diopiConstTensorHandle_t grad_output,
                                                   diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight,
                                                   diopiSize_t *bias_sizes,
                                                   diopiSize_t stride,
                                                   diopiSize_t padding,
                                                   diopiSize_t dilation,
                                                   bool transposed,
                                                   diopiSize_t output_padding,
                                                   int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto weight_tensor = DiopiTensor(weight);
    auto grad_output_tensor = DiopiTensor(grad_output);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_weight_tensor = DiopiTensor(grad_weight);

    diopiTensorHandle_t input_t;
    diopiTensorHandle_t weight_t;
    diopiTensorHandle_t grad_output_t;
    diopiTensorHandle_t grad_input_t;
    diopiTensorHandle_t grad_weight_t;

    auto permute_to_nhwc = [&](auto src, auto &dst) {
        std::vector<int64_t> axis{0, 2, 3, 1};
        auto src_tensor = DiopiTensor(src);
        std::vector<int64_t> src_shape_t_64(src_tensor.shape().size());
        for (int i = 0; i < src_tensor.shape().size(); ++i) {
            src_shape_t_64[i] = src_tensor.shape()[axis[i]];
        }
        diopiSize_t src_t_shape(src_shape_t_64.data(), src_shape_t_64.size());
        DIOPI_CALL(diopiRequireTensor(ctx, &dst, &src_t_shape, nullptr, src_tensor.dtype(), diopi_device));
        diopiSize_t nchw2nhwc(axis.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, dst, src, nchw2nhwc));
        return diopiSuccess;
    };

    DIOPI_CALL(permute_to_nhwc(input, input_t));
    DIOPI_CALL(permute_to_nhwc(weight, weight_t));
    DIOPI_CALL(permute_to_nhwc(grad_output, grad_output_t));
    DIOPI_CALL(permute_to_nhwc(grad_input, grad_input_t));
    DIOPI_CALL(permute_to_nhwc(grad_weight, grad_weight_t));

    auto input_tensor_t = DiopiTensor(input_t);
    auto weight_tensor_t = DiopiTensor(weight_t);
    auto grad_output_tensor_t = DiopiTensor(grad_output_t);
    auto grad_input_tensor_t = DiopiTensor(grad_input_t);
    auto grad_weight_tensor_t = DiopiTensor(grad_weight_t);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weight_desc(weight_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc output_grad_desc(grad_output_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc input_grad_desc(grad_input_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weight_grad_desc(grad_weight_tensor, CNNL_LAYOUT_NHWC);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> conv_desc;

    std::vector<int> stride_vec{stride.data, stride.data + stride.len};
    std::vector<int> padding_vec{padding.data, padding.data + padding.len};
    std::vector<int> dilation_vec{dilation.data, dilation.data + dilation.len};

    int padding_[4] = {padding_vec[0], padding_vec[1], padding_vec[0], padding_vec[1]};
    int stride_[2] = {stride_vec[0], stride_vec[1]};
    int dilation_[2] = {dilation_vec[0], dilation_vec[1]};

    cnnlDataType_t input_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&input_type, input_tensor_t.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(conv_desc.get(), 4, padding_, stride_, dilation_, groups, input_type));

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
                                                 input_tensor_t.data(),
                                                 output_grad_desc.get(),
                                                 grad_output_tensor_t.data(),
                                                 conv_desc.get(),
                                                 CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT,
                                                 workspace_filter,
                                                 workspace_size_filter,
                                                 NULL,
                                                 weight_grad_desc.get(),
                                                 grad_weight_tensor_t.data()));

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
                                               weight_tensor_t.data(),
                                               output_grad_desc.get(),
                                               grad_output_tensor_t.data(),
                                               conv_desc.get(),
                                               CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                               workspace_input,
                                               workspace_size_input,
                                               NULL,
                                               input_grad_desc.get(),
                                               grad_input_tensor_t.data()));

    std::vector<int64_t> perm_nhwc2nchw{0, 3, 1, 2};
    diopiSize_t nhwc2nchw(perm_nhwc2nchw.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, grad_input, grad_input_t, nhwc2nchw));
    DIOPI_CALL(diopiPermute(ctx, grad_weight, grad_weight_t, nhwc2nchw));

    if (grad3 != nullptr) {
        auto bias_grad_tensor = DiopiTensor(grad3);
        CnnlTensorDesc bias_grad_desc;
        DIOPI_CALL(bias_grad_desc.set(bias_grad_tensor, CNNL_LAYOUT_ARRAY));
        std::vector<int64_t> bias_shape{bias_grad_tensor.shape().begin(), bias_grad_tensor.shape().end()};
        bias_sizes->data = bias_shape.data();
        bias_sizes->len = bias_shape.size();
        size_t workspace_size_bias;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, output_grad_desc.get(), bias_grad_desc.get(), 3, &workspace_size_bias))

        void *workspace_bias = nullptr;
        if (0 != workspace_size_bias) {
            workspace_bias = requiresBuffer(ctx, workspace_size_bias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(handle,
                                              output_grad_desc.get(),
                                              grad_output_tensor_t.data(),
                                              3,
                                              bias_grad_desc.get(),
                                              bias_grad_tensor.data(),
                                              workspace_bias,
                                              workspace_size_bias));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
