/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_TORCH_EXT_KERNEL_H_
#define IMPL_TORCH_EXT_KERNEL_H_

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>

namespace ext {
namespace ops {

void apply_rotary_cuda(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& cos, const at::Tensor& sin, at::Tensor out1, at::Tensor out2,
                       const bool conj);

void rms_norm_forward(at::Tensor input, at::IntArrayRef normalized_shape, at::Tensor gamma, double epsilon, at::Tensor output, at::Tensor invvar);

void rms_norm_backward(at::Tensor dout, at::Tensor invvar, at::Tensor input, at::IntArrayRef normalized_shape, at::Tensor gamma, double epsilon,
                       at::Tensor grad_input, at::Tensor grad_gamma);

}  // namespace ops
}  // namespace ext
#endif  // IMPL_TORCH_EXT_KERNEL_H_
