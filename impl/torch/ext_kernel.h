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

void rms_norm_backward(Tensor dout, Tensor invvar, Tensor input, IntArrayRef normalized_shape, Tensor gamma, double epsilon, Tensor grad_input,
                       Tensor grad_gamma);

void apply_penalty_cuda(Tensor Logits, 
        Tensor presence_penalty, 
        Tensor frequency_penalty, 
        Tensor p_token_ids, 
        Tensor p_token_counts, 
        Tensor p_cumsum_seq_len, 
        int64_t stride_logit_b,
        int64_t stride_logit_s,
        int BLOCK_P
    )

}  // namespace ops
}  // namespace ext
#endif  // IMPL_TORCH_EXT_KERNEL_H_
