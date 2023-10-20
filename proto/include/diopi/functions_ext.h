/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj);

DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRMS, diopiConstTensorHandle_t input,
                                    diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps);

DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS,
                                            diopiSize_t normalized_shape, double eps);

// imitate  `_flash_attention_forward` in pytorch
// philox_seed
// philox_offset 是决定generator的种子，因此我们使用diopi generator来替代
// 估计这个softmax_logsumexp是指的
DIOPI_API diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, int64_t* max_q, int64_t* max_k,
                                               double dropout_p, bool is_causal, bool return_debug_mask, double* scale, diopiTensorHandle_t output,
                                               diopiTensorHandle_t softmax_logsumexp, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask);

DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                       diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                       diopiConstTensorHandle_t logsumexp, diopiConstTensorHandle_t cum_seq_q,
                                                       diopiConstTensorHandle_t cum_seq_k, int64_t* max_q, int64_t* max_k, double dropout_p, bool is_causal,
                                                       diopiGeneratorHandle_t gen, double* scale, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k,
                                                       diopiTensorHandle_t grad_v);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_MMCV_H_
