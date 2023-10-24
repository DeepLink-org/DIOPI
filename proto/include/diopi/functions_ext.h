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

/**
 * @brief Apply rotary embedding operation to an input tensor.
 * @param[in] ctx Context environment.
 * @param[out] out The output tensor containing the rotary embeddings. type = [float32, float16, float64].
 * @param[in] x The input tensor which rotary embedding will be applied. type = [float32, float16, float64].
 * @param[in] cos The cosine values. type = [float32, float16, float64].
 * @param[in] sin The sine values. type = [float32, float16, float64].
 * @param[in] conj bool: If `true`, computes the complex conjugate of the rotary embeddings for forward.If `false`, computes regular rotary embeddings for
 * backward.
 * @param[in] interleaved bool:
 *   - When set to `false`, rotary embedding is applied by splitting 'x' in half and separately applying sine and cosine to each half.
 *   - When set to `true`, rotary embedding is applied by pairing every two elements in 'x' and applying sine and cosine to each pair.
 */
DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj, const bool interleaved);

/**
 * @brief Apply Root Mean Square (RMS) Normalization to the input tensor.
 * @param[in] ctx Context environment.
 * @param[out] out the output tensor containing the normalized values. type = [float32, float16, float64].
 * @param[in] invRMS The tensor containing the inverse of root mean square. type = [float32, float16, float64].
 * @param[in] input The input tensor to be normalized. type = [float32, float16, float64].
 * @param[in] normalized_shape The shape of the normalization.
 * @param[in] weight The gain parameter used to re-scale the standardized summed inputs type = [float32, float16, float64].
 * @param[in] bias The bias tensor for the normalization. type = [float32, float16, float64].
 * @param[in] eps A small value to avoid division by zero. type = [float64].
 */
DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRMS, diopiConstTensorHandle_t input,
                                    diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps);

/**
 * @brief Compute the backward pass for Root Mean Square (RMS) Normalization.
 * @param[in] ctx Context environment.
 * @param[out] gradInput The gradient of the input tensor. type = [float32, float16, float64].
 * @param[out] gradWeight The gradient of the weight parameter. type = [float32, float16, float64].
 * @param[out] gradBias The gradient of the bias parameter. type = [float32, float16, float64].
 * @param[in] gradOutput The gradient of the output from the forward pass. type = [float32, float16, float64].
 * @param[in] input The input tensor used in the forward pass. type = [float32, float16, float64].
 * @param[in] weight The weight parameter used in the forward pass. type = [float32, float16, float64].
 * @param[in] bias The bias used in the forward pass. type = [float32, float16, float64].
 * @param[in] invRMS The inverse of the root mean square values computed in the forward pass. type = [float32, float16, float64].
 * @param[in] normalized_shape The shape of the normalization.
 * @param[in] eps A small value used in the computation to avoid division by zero. type = [float64].
 */
DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS,
                                            diopiSize_t normalized_shape, double eps);

/**
 * @brief Compute the forward pass for MultiheadAttention.
 * @param[in] ctx Context environment.
 * @param[in] q Query tensor. type = [float32, float16, float64].
 *   - for unpaded: shape = [q_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, q_seq_len, head_num, head_dim]
 * @param[in] k Key tensor. type = [float32, float16, float64].
 *   - for unpaded: shape = [k_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, k_seq_len, head_num, head_dim]
 * @param[in] v Value tensor. type = [float32, float16, float64].
 *   - for unpaded: shape = [v_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, v_seq_len, head_num, head_dim]
 * @param[in] cum_seq_q Cumulative sequence length for the query. type = [int64, int32].
 *   - for unpaded: shape = [batch_size+1, ]
 *   - for padded: nullptr
 * @param[in] cum_seq_k Cumulative sequence length for the key. type = [int64, int32].
 *   - for unpaded: shape = [batch_size+1, ]
 *   - for padded: nullptr
 * @param[in] max_q Maximum sequence length for the query. For tensors already padded, pass nullptr.   type = [int64].
 * @param[in] max_k Maximum sequence length for the key. For tensors already padded, pass nullptr.  type = [int64].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] return_debug_mask Flag indicating if the attention debug mask should be returned. type = [bool].
 * @param[in] scale Scaling factor for attention weights. type = [float32, float16, float64].
 * @param[out] out Tensor containing the result after applying multi-head attention. type = [float32, float16, float64].
 *   - for unpaded: shape = [q_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, q_seq_len, head_num, head_dim]
 * @param[out] softmax_lse Tensor representing the log-sum-exp of the softmax values. type = [float32, float16, float64].
 *   - for unpaded: shape = [batch_size, head_num, max_q]
 *   - for padded: shape = [batch_size, head_num, q_seq_len]
 * @param[out] gen Handle for the random number generator used in dropout.
 * @param[out] debug_attn_mask Debugging tensor for the attention mask (returned if return_debug_mask is true). type = [bool].
 *   - for unpadded: shape = [batch_size, num_heads, max_q, max_k]
 *   - for padded: shape = [batch_size, num_heads, q_seq_len, k_seq_len]
 */
DIOPI_API diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, const int64_t* max_q,
                                               const int64_t* max_k, double dropout_p, bool is_causal, bool return_debug_mask, double* scale,
                                               diopiTensorHandle_t out, diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen,
                                               diopiTensorHandle_t debug_attn_mask);

/**
 * @brief Compute the forward pass for MultiheadAttention.
 * @param[in] ctx Context environment.
 * @param[in] grad_out The gradient of the output tensor.
 *   - for unpaded: shape = [q_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, q_seq_len, head_num, head_dim]
 * @param[in] q Query tensor from the forward pass. type = [float32, float16, float64].
 *   - for unpaded: shape = [q_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, q_seq_len, head_num, head_dim]
 * @param[in] k Key tensor from the forward pass. type = [float32, float16, float64].
 *   - for unpaded: shape = [k_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, k_seq_len, head_num, head_dim]
 * @param[in] v Value tensor from the forward pass. type = [float32, float16, float64].
 *   - for unpaded: shape = [v_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, v_seq_len, head_num, head_dim]
 * @param[in] out Output tensor from the forward pass.  type = [float32, float16, float64].
 *   - for unpaded: shape = [q_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, q_seq_len, head_num, head_dim]
 * @param[in] softmax_lse Tensor representing the log-sum-exp of softmax values from the forward pass. type = [float32, float16, float64].
 *   - for unpaded: shape = [batch_size, head_num, max_q]
 *   - for padded: shape = [batch_size, head_num, q_seq_len]
 * @param[in] cum_seq_q Cumulative sequence length for the query. type = [int64, int32].
 *   - for unpaded: shape = [batch_size+1, ]
 *   - for padded: nullptr
 * @param[in] cum_seq_k Cumulative sequence length for the key. type = [int64, int32].
 *   - for unpaded: shape = [batch_size+1, ]
 *   - for padded: nullptr
 * @param[in] max_q Maximum sequence length for the query. For tensors already padded, pass nullptr.   type = [int64].
 * @param[in] max_k Maximum sequence length for the key. For tensors already padded, pass nullptr.  type = [int64].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] return_debug_mask Flag from the forward pass indicating if the attention was causal (masking future tokens). type = [bool].
 *   - for unpadded: shape = [batch_size, num_heads, max_q, max_k]
 *   - for padded: shape = [batch_size, num_heads, q_seq_len, k_seq_len]
 * @param[in] gen Handle representing the random number generator used for dropout in the forward pass.
 * @param[in] scale Scaling factor used for attention weights in the forward pass. type = [float32, float16, float64].
 * @param[out] grad_q The gradient of the query tensor. type = [float32, float16, float64].
 *   - for unpaded: shape = [q_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, q_seq_len, head_num, head_dim]
 * @param[out] grad_k The gradient of the key tensor. type = [float32, float16, float64].
 *   - for unpaded: shape = [k_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, k_seq_len, head_num, head_dim]
 * @param[out] grad_v The gradient of the value tensor. type = [float32, float16, float64].
 *   - for unpaded: shape = [v_nums, head_num, head_dim]
 *   - for padded: shape = [batch_size, v_seq_len, head_num, head_dim]
 */
DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                                       diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                       diopiConstTensorHandle_t softmax_lse, diopiConstTensorHandle_t cum_seq_q,
                                                       diopiConstTensorHandle_t cum_seq_k, const int64_t* max_q, const int64_t* max_k, double dropout_p,
                                                       bool is_causal, diopiGeneratorHandle_t gen, double* scale, diopiTensorHandle_t grad_q,
                                                       diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_
