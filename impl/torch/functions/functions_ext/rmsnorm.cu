#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

namespace ext {
namespace ops {
namespace {

#define DIOPI_TORCH_EXT_CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define DIOPI_TORCH_EXT_CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define DIOPI_TORCH_EXT_CHECK_INPUT(x) \
    DIOPI_TORCH_EXT_CHECK_CUDA(x);     \
    DIOPI_TORCH_EXT_CHECK_CONTIGUOUS(x)

void compute_n1_n2(at::Tensor input, at::IntArrayRef normalized_shape, int& n1, int& n2) {
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0; i < static_cast<int>(normalized_shape.size()); ++i) {
        assert(input.sizes()[i + idiff] == normalized_shape[i]);
        n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0; i < idiff; ++i) {
        n1 *= input.sizes()[i];
    }
}

void check_args(at::IntArrayRef normalized_shape, at::Tensor gamma, at::Tensor beta) {
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
    TORCH_CHECK(!beta.defined() || beta.sizes().equals(normalized_shape));
}

void check_args(at::IntArrayRef normalized_shape, at::Tensor gamma) { TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape)); }

void check_args(at::Tensor input, at::IntArrayRef normalized_shape, int& n1, int& n2) {
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
        std::stringstream ss;
        ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
           << "containing at least one element, but got normalized_shape=" << normalized_shape;
        throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim || !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
        std::stringstream ss;
        ss << "Given normalized_shape=" << normalized_shape << ", expected input with shape [*";
        for (auto size : normalized_shape) {
            ss << ", " << size;
        }
        ss << "], but got input of size" << input_shape;
        throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input, normalized_shape, n1, n2);
}

void check_args(at::Tensor input, at::IntArrayRef normalized_shape, at::Tensor gamma, at::Tensor beta, int& n1, int& n2) {
    check_args(input, normalized_shape, n1, n2);
    check_args(normalized_shape, gamma, beta);
}

void check_args(at::Tensor input, at::IntArrayRef normalized_shape, at::Tensor gamma, int& n1, int& n2) {
    check_args(input, normalized_shape, n1, n2);
    check_args(normalized_shape, gamma);
}

}  // namespace

// implemented in layer_norm_cuda_kernel.cu
void cuda_rms_norm(at::Tensor* output, at::Tensor* invvar, at::Tensor* input, int n1, int n2, at::IntArrayRef normalized_shape, at::Tensor* gamma,
                   double epsilon);

// implemented in layer_norm_cuda_kernel.cu
void cuda_rms_norm_gradient(at::Tensor* dout, at::Tensor* invvar, at::Tensor* input, int n1, int n2, at::IntArrayRef normalized_shape, at::Tensor* gamma,
                            double epsilon, at::Tensor* grad_input, at::Tensor* grad_gamma);

// 供cpp层调用函数，向外暴露的函数

void rms_norm_forward(at::Tensor input, at::IntArrayRef normalized_shape, at::Tensor gamma, double epsilon, at::Tensor output, at::Tensor invvar) {
    DIOPI_TORCH_EXT_CHECK_INPUT(input);
    DIOPI_TORCH_EXT_CHECK_INPUT(gamma);
    int n1;
    int n2;
    check_args(input, normalized_shape, gamma, n1, n2);
    const auto stats_dtype =
        (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
    cuda_rms_norm(&output, &invvar, &input, n1, n2, normalized_shape, &gamma, epsilon);
    return;
}

void rms_norm_backward(at::Tensor dout, at::Tensor invvar, at::Tensor input, at::IntArrayRef normalized_shape, at::Tensor gamma, double epsilon,
                       at::Tensor grad_input, at::Tensor grad_gamma) {
    DIOPI_TORCH_EXT_CHECK_INPUT(dout);
    DIOPI_TORCH_EXT_CHECK_INPUT(invvar);
    DIOPI_TORCH_EXT_CHECK_INPUT(input);
    DIOPI_TORCH_EXT_CHECK_INPUT(gamma);
    int n1;
    int n2;
    check_args(input, normalized_shape, gamma, n1, n2);
    cuda_rms_norm_gradient(&dout, &invvar, &input, n1, n2, normalized_shape, &gamma, epsilon, &grad_input, &grad_gamma);
    return;
}

}  // namespace ops
}  // namespace ext
