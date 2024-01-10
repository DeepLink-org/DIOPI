#include <diopi/functions.h>
#include <math.h>

#include <cstring>

#include "../common/common.hpp"
#include "xdnn_pytorch/xdnn_pytorch.h"

#define FLT_MIN __FLT_MIN__

namespace impl {
namespace kunlunxin {

static const char* name = "KLXDevice";
static char version[1024] = {0};

DIOPI_RT_API const char* diopiGetVendorName() { return name; }

DIOPI_RT_API const char* diopiGetImplVersion() {
    int klx_version = 1;
    if (strlen(version) == 0) {
        const char* diopiVersion = diopiGetVersion();
        sprintf(version, "KLX Version: %d; %s", klx_version, diopiVersion);
    }
    return version;
}

DIOPI_API diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _in = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    DIOPI_CALL_XDNN(xdnn_pytorch::cast(ctx_xpu, _in, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);

    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);
    xdnn_pytorch::Tensor _in = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Scalar _scalar = impl::kunlunxin::build_xtorch_scalar(value);
    DIOPI_CALL_XDNN(xdnn_pytorch::fill__scalar(ctx_xpu, _in, _scalar, _in));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, diopiGeneratorHandle_t gen) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _inout = impl::kunlunxin::build_xtorch_tensor(inout);
    int64_t _to = to != nullptr ? *to : 0x7fffffff;
    xdnn_pytorch::Generator generator;

    DIOPI_CALL_XDNN(xdnn_pytorch::random__from(ctx_xpu, _inout, from, _to, generator));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, diopiGeneratorHandle_t generator) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);

    DIOPI_CALL_XDNN(xdnn_pytorch::randperm(ctx_xpu, _out, n));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _src = impl::kunlunxin::build_xtorch_tensor(src);
    xdnn_pytorch::Tensor _dest = impl::kunlunxin::build_xtorch_tensor(dest);

    DIOPI_CALL_XDNN(xdnn_pytorch::_to_copy(ctx_xpu,
                                           _src,
                                           xdnn_pytorch::optional<int64_t>(),
                                           xdnn_pytorch::optional<int64_t>(),
                                           xdnn_pytorch::optional<xdnn_pytorch::Device>(),
                                           xdnn_pytorch::optional<bool>(),
                                           false,
                                           xdnn_pytorch::optional<int64_t>(),
                                           _dest));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum,
                                double dampening, double weight_decay, bool nesterov) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _w = impl::kunlunxin::build_xtorch_tensor(w);
    xdnn_pytorch::Tensor _dw = impl::kunlunxin::build_xtorch_tensor(dw);
    xdnn_pytorch::Tensor _buf = impl::kunlunxin::build_xtorch_tensor(buf);

    DIOPI_CALL_XDNN(xdnn_pytorch::sgd(ctx_xpu, _w, _dw, _buf, lr, momentum, dampening, weight_decay, nesterov));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _in = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _bias = impl::kunlunxin::build_xtorch_tensor(bias);
    xtorch_vec _stride = impl::kunlunxin::build_xtorch_vec(stride);
    xtorch_vec _pad = impl::kunlunxin::build_xtorch_vec(padding);
    xtorch_vec _dilation = impl::kunlunxin::build_xtorch_vec(dilation);

    DIOPI_CALL_XDNN(xdnn_pytorch::convolution(ctx_xpu, _in, _weight, _bias, _stride, _pad, _dilation, false, {}, groups, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, int64_t groups) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_weight = impl::kunlunxin::build_xtorch_tensor(grad_weight);
    xdnn_pytorch::Tensor _grad_bias = impl::kunlunxin::build_xtorch_tensor(grad3);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);
    xtorch_vec _bias_sizes;
    xtorch_vec _output_padding;
    if (bias_sizes) {
        _bias_sizes = impl::kunlunxin::build_xtorch_vec(*bias_sizes);
    }
    xtorch_vec _stride = impl::kunlunxin::build_xtorch_vec(stride);
    xtorch_vec _padding = impl::kunlunxin::build_xtorch_vec(padding);
    xtorch_vec _dilation = impl::kunlunxin::build_xtorch_vec(dilation);
    auto output_mask = std::array<bool, 3>{true, true, false};
    DIOPI_CALL_XDNN(xdnn_pytorch::convolution_backward(ctx_xpu,
                                                       _grad_output,
                                                       _input,
                                                       _weight,
                                                       _bias_sizes,
                                                       _stride,
                                                       _padding,
                                                       _dilation,
                                                       false,  // transposed,
                                                       _output_padding,
                                                       groups,
                                                       output_mask,
                                                       _grad_input,
                                                       _grad_weight,
                                                       _grad_bias));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _other = impl::kunlunxin::build_xtorch_tensor(other);
    xdnn_pytorch::Scalar _alpha = impl::kunlunxin::build_xtorch_scalar(alpha);
    DIOPI_CALL_XDNN(xdnn_pytorch::add_tensor(ctx_xpu, _input, _other, _alpha, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    impl::kunlunxin::diopiAdd(ctx, input, input, other, alpha);
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Scalar _other = impl::kunlunxin::build_xtorch_scalar(other);
    xdnn_pytorch::Scalar _alpha = impl::kunlunxin::build_xtorch_scalar(alpha);
    DIOPI_CALL_XDNN(xdnn_pytorch::add_scalar(ctx_xpu, _input, _other, _alpha, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    impl::kunlunxin::diopiAddScalar(ctx, input, input, other, alpha);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _save_mean = impl::kunlunxin::build_xtorch_tensor(save_mean);
    xdnn_pytorch::Tensor _save_invstd = impl::kunlunxin::build_xtorch_tensor(save_invstd);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);
    xdnn_pytorch::Tensor _bias = impl::kunlunxin::build_xtorch_tensor(bias);
    xdnn_pytorch::Tensor _running_mean = impl::kunlunxin::build_xtorch_tensor(running_mean);
    xdnn_pytorch::Tensor _running_var = impl::kunlunxin::build_xtorch_tensor(running_var);
    DIOPI_CALL_XDNN(
        xdnn_pytorch::native_batch_norm(ctx_xpu, _input, _weight, _bias, _running_mean, _running_var, training, momentum, eps, _out, _save_mean, _save_invstd));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t rounding_mode) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _other = impl::kunlunxin::build_xtorch_tensor(other);
    std::string mode = rounding_mode == RoundModeFloor ? "floor" : rounding_mode == RoundModeTrunc ? "trunc" : "";
    if (mode == "") {
        DIOPI_CALL_XDNN(xdnn_pytorch::div_tensor_mode(ctx_xpu, _input, _other, xdnn_pytorch::optional<xdnn_pytorch::StringView>(), _out))
    } else {
        DIOPI_CALL_XDNN(xdnn_pytorch::div_tensor_mode(ctx_xpu, _input, _other, xdnn_pytorch::optional<xdnn_pytorch::StringView>({mode.c_str()}), _out))
    }
    DIOPI_CALL_XDNN(xdnn_pytorch::div_tensor(ctx_xpu, _input, _other, _out))
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _other = impl::kunlunxin::build_xtorch_tensor(other);
    DIOPI_CALL_XDNN(xdnn_pytorch::mul_tensor(ctx_xpu, _input, _other, _out))
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      diopiRoundMode_t rounding_mode) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Scalar _other = impl::kunlunxin::build_xtorch_scalar(other);
    DIOPI_CALL_XDNN(xdnn_pytorch::div_scalar(ctx_xpu, _input, _other, _out))
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Scalar _other = impl::kunlunxin::build_xtorch_scalar(other);
    DIOPI_CALL_XDNN(xdnn_pytorch::mul_scalar(ctx_xpu, _input, _other, _out))
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Scalar _other = impl::kunlunxin::build_xtorch_scalar(other);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(input);
    DIOPI_CALL_XDNN(xdnn_pytorch::mul_scalar(ctx_xpu, _input, _other, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var,
                                              diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_weight = impl::kunlunxin::build_xtorch_tensor(grad_weight);
    xdnn_pytorch::Tensor _grad_bias = impl::kunlunxin::build_xtorch_tensor(grad_bias);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);
    xdnn_pytorch::Tensor _running_mean = impl::kunlunxin::build_xtorch_tensor(running_mean);
    xdnn_pytorch::Tensor _running_var = impl::kunlunxin::build_xtorch_tensor(running_var);
    xdnn_pytorch::Tensor _save_mean = impl::kunlunxin::build_xtorch_tensor(save_mean);
    xdnn_pytorch::Tensor _save_invstd = impl::kunlunxin::build_xtorch_tensor(save_invstd);
    auto output_mask = std::array<bool, 3>{true, true, true};
    DIOPI_CALL_XDNN(xdnn_pytorch::native_batch_norm_backward(ctx_xpu,
                                                             _grad_output,
                                                             _input,
                                                             _weight,
                                                             _running_mean,
                                                             _running_var,
                                                             _save_mean,
                                                             _save_invstd,
                                                             training,
                                                             eps,
                                                             output_mask,
                                                             _grad_input,
                                                             _grad_weight,
                                                             _grad_bias));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    DIOPI_CALL_XDNN(xdnn_pytorch::relu(ctx_xpu, _input, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    DIOPI_CALL_XDNN(xdnn_pytorch::relu(ctx_xpu, _input, _input));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _in = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xtorch_vec _output_size = impl::kunlunxin::build_xtorch_vec(output_size);

    DIOPI_CALL_XDNN(xdnn_pytorch::_adaptive_avg_pool2d(ctx_xpu, _in, _output_size, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_in = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_out = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _in = impl::kunlunxin::build_xtorch_tensor(input);

    DIOPI_CALL_XDNN(xdnn_pytorch::_adaptive_avg_pool2d_backward(ctx_xpu, _grad_out, _in, _grad_in));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                   diopiConstTensorHandle_t bias) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);
    xdnn_pytorch::Tensor _bias = impl::kunlunxin::build_xtorch_tensor(bias);

    DIOPI_CALL_XDNN(xdnn_pytorch::linear(ctx_xpu, _input, _weight, _bias, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                           diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_weight = impl::kunlunxin::build_xtorch_tensor(grad_weight);
    xdnn_pytorch::Tensor _grad_bias = impl::kunlunxin::build_xtorch_tensor(grad_bias);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);

    DIOPI_CALL_XDNN(xdnn_pytorch::linear_backward(ctx_xpu, _input, _grad_output, _weight, _grad_input, _grad_weight, _grad_bias));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                                  diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _mat1 = impl::kunlunxin::build_xtorch_tensor(mat1);
    xdnn_pytorch::Tensor _mat2 = impl::kunlunxin::build_xtorch_tensor(mat2);
    xdnn_pytorch::Scalar _beta = impl::kunlunxin::build_xtorch_scalar(beta);
    xdnn_pytorch::Scalar _alpha = impl::kunlunxin::build_xtorch_scalar(alpha);

    DIOPI_CALL_XDNN(xdnn_pytorch::addmm(ctx_xpu, _input, _mat1, _mat2, _beta, _alpha, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                             diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _target = impl::kunlunxin::build_xtorch_tensor(target);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);

    DIOPI_CALL_XDNN(xdnn_pytorch::cross_entropy_loss(ctx_xpu, _input, _target, _weight, (int64_t)reduction, ignore_index, label_smoothing, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _target = impl::kunlunxin::build_xtorch_tensor(target);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);

    DIOPI_CALL_XDNN(xdnn_pytorch::cross_entropy_loss_backward(
        ctx_xpu, _grad_output, _input, _target, _weight, (int64_t)reduction, ignore_index, label_smoothing, _grad_input));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _inout = impl::kunlunxin::build_xtorch_tensor(inout);
    xdnn_pytorch::Tensor _ret = impl::kunlunxin::build_xtorch_tensor(inout);
    xdnn_pytorch::Generator _gen;
    DIOPI_CALL_XDNN(xdnn_pytorch::normal_(ctx_xpu, _inout, mean, std, xdnn_pytorch::optional<xdnn_pytorch::Generator>(_gen), _ret));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::ScalarType _dtype = xdnn_pytorch::ScalarType::kfloat32;

    DIOPI_CALL_XDNN(xdnn_pytorch::log_softmax(ctx_xpu, _input, dim, _dtype, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _output = impl::kunlunxin::build_xtorch_tensor(output);
    // xdnn_pytorch::ScalarType _input_dtype = impl::kunlunxin::get_xtorch_type(input_dtype);
    xdnn_pytorch::ScalarType _input_dtype = xdnn_pytorch::ScalarType::kfloat32;

    DIOPI_CALL_XDNN(xdnn_pytorch::_log_softmax_backward_data(ctx_xpu, _grad_output, _output, dim, _input_dtype, _grad_input));
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _target = impl::kunlunxin::build_xtorch_tensor(target);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);

    // ignore_index = -100; // wait for api.so update
    DIOPI_CALL_XDNN(xdnn_pytorch::nll_loss(ctx_xpu, _input, _target, _weight, (int64_t)reduction, ignore_index, _out));
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _target = impl::kunlunxin::build_xtorch_tensor(target);
    xdnn_pytorch::Tensor _weight = impl::kunlunxin::build_xtorch_tensor(weight);

    xdnn_pytorch::Tensor _total_weight = {{0}, {0}, xdnn_pytorch::ScalarType::kfloat32, nullptr};

    // ignore_index = -100; // wait for api.so update
    DIOPI_CALL_XDNN(
        xdnn_pytorch::nll_loss_backward(ctx_xpu, _grad_output, _input, _target, _weight, (int64_t)reduction, ignore_index, _total_weight, _grad_input));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);

    if (dim.len == 0) {
        DIOPI_CALL_XDNN(xdnn_pytorch::sum(ctx_xpu, _input, _out.type, _out));
    } else {
        xtorch_vec _dim = impl::kunlunxin::build_xtorch_vec(dim);
        DIOPI_CALL_XDNN(xdnn_pytorch::sum_dim_IntList(ctx_xpu, _input, _dim, false, _out.type, _out));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);

    if (dim.len == 0) {
        xdnn_pytorch::Tensor _inputTemp = impl::kunlunxin::build_xtorch_tensor(input);
        DIOPI_CALL_XDNN(xdnn_pytorch::mean(ctx_xpu, _input, _out.type, _out));
    } else {
        xtorch_vec _dim = impl::kunlunxin::build_xtorch_vec(dim);
        DIOPI_CALL_XDNN(xdnn_pytorch::mean_dim(ctx_xpu, _input, _dim, false, _out.type, _out));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                                 bool descending, const bool* stable) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _values = impl::kunlunxin::build_xtorch_tensor(values);
    xdnn_pytorch::Tensor _indices = impl::kunlunxin::build_xtorch_tensor(indices);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);

    bool _stable = (stable == nullptr) ? false : (*stable);
    DIOPI_CALL_XDNN(xdnn_pytorch::sort_stable(ctx_xpu, _input, dim, descending, _stable, _values, _indices));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Tensor _other = impl::kunlunxin::build_xtorch_tensor(other);
    xdnn_pytorch::Scalar _alpha = impl::kunlunxin::build_xtorch_scalar(alpha);
    DIOPI_CALL_XDNN(xdnn_pytorch::sub_tensor(ctx_xpu, _input, _other, _alpha, _out));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _grad_input = impl::kunlunxin::build_xtorch_tensor(grad_input);
    xdnn_pytorch::Tensor _grad_output = impl::kunlunxin::build_xtorch_tensor(grad_output);
    xdnn_pytorch::Tensor _input = impl::kunlunxin::build_xtorch_tensor(input);
    xdnn_pytorch::Scalar _threshold = impl::kunlunxin::build_xtorch_scalar(threshold);
    DIOPI_CALL_XDNN(xdnn_pytorch::threshold_backward(ctx_xpu, _grad_output, _input, _threshold, _grad_input));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim) {
    xdnn::Context* ctx_xpu = impl::kunlunxin::set_cur_ctx(ctx);
    xdnn_pytorch::Tensor _out = impl::kunlunxin::build_xtorch_tensor(out);
    auto _inputs = impl::kunlunxin::build_xtorch_tensorlist(tensors, num_inputs);

    DIOPI_CALL_XDNN(xdnn_pytorch::cat(ctx_xpu, _inputs, dim, _out));
    return diopiSuccess;
}

}  // namespace kunlunxin
}  // namespace impl
