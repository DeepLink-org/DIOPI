#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

namespace {
enum class Transform {
    NONE,
    NCHW_TO_NHWC,
};

template <typename T, typename SRC>
diopiError_t convert_vector(const SRC &data, std::vector<T> &out, Transform trans = Transform::NONE) {
    if (Transform::NONE == trans) {
        out = std::vector<T>{data.begin(), data.end()};
        return diopiSuccess;
    } else if (Transform::NCHW_TO_NHWC == trans) {
        out.resize(4);
        out[0] = data[0];
        out[1] = data[2];
        out[2] = data[3];
        out[3] = data[1];
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}
}  // namespace

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

    auto input_tensor = makeTensor(input);
    auto output_tensor = makeTensor(out);
    auto weight_tensor = makeTensor(weight);

    std::vector<int64_t> input_shape_t_64;
    std::vector<int64_t> weight_shape_t_64;
    std::vector<int64_t> output_shape_t_64;

    DIOPI_CALL(convert_vector(input_tensor.shape(), input_shape_t_64, Transform::NCHW_TO_NHWC));
    DIOPI_CALL(convert_vector(weight_tensor.shape(), weight_shape_t_64, Transform::NCHW_TO_NHWC));
    DIOPI_CALL(convert_vector(output_tensor.shape(), output_shape_t_64, Transform::NCHW_TO_NHWC));

    std::vector<int> input_shape_t{input_shape_t_64.begin(), input_shape_t_64.end()};
    std::vector<int> weight_shape_t{weight_shape_t_64.begin(), weight_shape_t_64.end()};
    std::vector<int> output_shape_t{output_shape_t_64.begin(), output_shape_t_64.end()};

    diopiTensorHandle_t input_tensor_t;
    diopiTensorHandle_t weight_tensor_t;
    diopiTensorHandle_t output_tensor_t;

    diopiSize_t input_t_shape(input_shape_t_64.data(), input_shape_t_64.size());
    diopiSize_t weight_t_shape(weight_shape_t_64.data(), weight_shape_t_64.size());
    diopiSize_t output_t_shape(output_shape_t_64.data(), output_shape_t_64.size());

    DIOPI_CALL(diopiRequireTensor(ctx, &input_tensor_t, &input_t_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiRequireTensor(ctx, &weight_tensor_t, &weight_t_shape, nullptr, weight_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiRequireTensor(ctx, &output_tensor_t, &output_t_shape, nullptr, output_tensor.dtype(), diopi_device));

    std::vector<int64_t> perm_nchw2nhwc{0, 2, 3, 1};
    diopiSize_t nchw2nhwc(perm_nchw2nhwc.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, input_tensor_t, input, nchw2nhwc));
    DIOPI_CALL(diopiPermute(ctx, weight_tensor_t, weight, nchw2nhwc));

    CnnlTensorDesc input_desc, output_desc, weight_desc;
    DIOPI_CALL(input_desc.set(input_tensor, CNNL_LAYOUT_NHWC, input_shape_t));
    DIOPI_CALL(output_desc.set(output_tensor, CNNL_LAYOUT_NHWC, output_shape_t));
    DIOPI_CALL(weight_desc.set(weight_tensor, CNNL_LAYOUT_NHWC, weight_shape_t));

    const void *bias_ptr = nullptr;
    CnnlTensorDesc bias_desc;
    if (nullptr != bias) {
        auto bias_tensor = makeTensor(bias);
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
    DIOPI_CALL(convertType(&input_type, input_tensor.dtype()));
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
                                          makeTensor(input_tensor_t).data(),
                                          weight_desc.get(),
                                          makeTensor(weight_tensor_t).data(),
                                          bias_desc.get(),
                                          bias_ptr,
                                          workspace,
                                          workspace_size,
                                          NULL,
                                          output_desc.get(),
                                          makeTensor(output_tensor_t).data()));

    std::vector<int64_t> perm_nhwc2nchw{0, 3, 1, 2};
    diopiSize_t nhwc2nchw(perm_nhwc2nchw.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, out, output_tensor_t, nhwc2nchw));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
