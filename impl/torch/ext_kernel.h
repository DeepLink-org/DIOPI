/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_TORCH_EXT_KERNEL_H_
#define IMPL_TORCH_EXT_KERNEL_H_

#include <ATen/ATen.h>

#include <vector>

namespace ext {
namespace ops {

using namespace at;

void apply_rotary_cuda(const Tensor x1, const Tensor x2, const Tensor cos, const Tensor sin, Tensor out1, Tensor out2, const bool conj);

void rms_norm_forward(Tensor input, IntArrayRef normalized_shape, Tensor gamma, double epsilon, Tensor output, Tensor invvar);

void rms_norm_backward(Tensor dout, Tensor invvar, Tensor input, IntArrayRef normalized_shape, Tensor gamma, double epsilon, Tensor grad_input, Tensor grad_gamma);

}  // namespace ops
}  // namespace ext
#endif  // IMPL_TORCH_EXT_KERNEL_H_