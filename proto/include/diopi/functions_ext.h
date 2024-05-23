/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_

#include <diopi/diopirt.h>

#include <cstdint>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/**
 * @brief Apply rotary embedding operation to an input tensor.
 * @param[in] ctx The diopi context.
 * @param[out] out The output tensor containing the rotary embeddings. type = [bfloat16, float16, float32, float64].
 * @param[in] x The input tensor which rotary embedding will be applied. type = [bfloat16, float16, float32, float64].
 * @param[in] cos The cosine values. type = [bfloat16, float16, float32, float64].
 * @param[in] sin The sine values. type = [bfloat16, float16, float32, float64].
 * @param[in] conj bool: If `false`, compute rotary embeddings for forward. If `true`, computes the backward of rotary embeddings according to the conjugate of
 * the rotary matrix.
 * @param[in] interleaved bool:
 *   - When set to `false`, rotary embedding is applied by splitting 'x' in half and separately applying sine and cosine to each half.
 *   - When set to `true`, rotary embedding is applied by pairing every two elements in 'x' and applying sine and cosine to each pair.
 */
DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj, const bool interleaved);

/**
 * @brief Apply Root Mean Square (RMS) Normalization to the input tensor.
 * @param[in] ctx The diopi context.
 * @param[out] out The output tensor containing the normalized values. type = [bfloat16, float16, float32, float64].
 * @param[out] inv_rms The tensor containing the inverse of root mean square. type = [float32, float64].
 * @param[in] input The input tensor to be normalized. type = [bfloat16, float16, float32, float64].
 * @param[in] normalized_shape The shape of the partial input which is needed to be normalized.
 * @param[in] weight The gain parameter used to re-scale the standardized summed inputs type = [bfloat16, float16, float32, float64].
 * @param[in] bias The bias tensor for the normalization. type = [bfloat16, float16, float32, float64].
 * @param[in] eps A small value to avoid division by zero.
 */
DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t inv_rms, diopiConstTensorHandle_t input,
                                    diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps);

/**
 * @brief Compute the backward pass for Root Mean Square (RMS) Normalization.
 * @param[in] ctx The diopi context.
 * @param[out] grad_input The gradient of the input tensor. type = [bfloat16, float16, float32, float64].
 * @param[out] grad_weight The gradient of the weight parameter. type = [bfloat16, float16, float32, float64].
 * @param[out] grad_bias The gradient of the bias parameter. type = [bfloat16, float16, float32, float64].
 * @param[in] grad_output The gradient of the output from the forward pass. type = [bfloat16, float16, float32, float64].
 * @param[in] input The input tensor used in the forward pass. type = [bfloat16, float16, float32, float64].
 * @param[in] weight The weight parameter used in the forward pass. type = [bfloat16, float16, float32, float64].
 * @param[in] bias The bias used in the forward pass. type = [bfloat16, float16, float32, float64].
 * @param[in] inv_rms The inverse of the root mean square values computed in the forward pass. type = [bfloat16, float32, float64].
 * @param[in] normalized_shape The shape of the partial input which is needed to be normalized.
 * @param[in] eps A small value used in the computation to avoid division by zero.
 */
DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                            diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t inv_rms,
                                            diopiSize_t normalized_shape, double eps);

// This interface has been deprecated, please use a more suitable diopiFlashAttention series interface.
/**
 * @brief Compute the forward pass for MultiheadAttention.
 * @param[in] ctx The diopi context.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 *        - For the implementation of flash-attention in CUDA, it is necessary to pad for 'q' to ensure that the 'head_dim' of the output from 'q' is
 * divisible by 8. Therefore, it is required to perform in-place modifications on 'q' by setting its data type to 'diopiTensorHandle_t'.
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 *        - For the implementation of flash-attention in CUDA, it is necessary to pad for 'k' to ensure that the 'head_dim' of the output from 'k' is
 * divisible by 8. Therefore, it is required to perform in-place modifications on 'k' by setting its data type to 'diopiTensorHandle_t'.
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 *        - For the implementation of flash-attention in CUDA, it is necessary to pad for 'v' to ensure that the 'head_dim' of the output from 'v' is
 * divisible by 8. Therefore, it is required to perform in-place modifications on 'v' by setting its data type to 'diopiTensorHandle_t'.
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] return_debug_mask Flag indicating if the attention debug mask should be returned. type = [bool].
 * @param[in] scale Scaling factor for attention weights. type = [float32, float16, float64].
 * @param[out] out Tensor containing the result after applying multi-head attention. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32,
 * float16, float64].
 * @param[out] softmax_lse Tensor representing the log-sum-exp of the softmax values. shape = [batch_size, head_num, q_seq_len]. type = [float32, float16,
 * float64].
 * @param[out] gen Handle for the random number generator used in dropout.
 * @param[out] debug_attn_mask Debugging tensor for the attention mask (returned if return_debug_mask is true). shape = [batch_size, num_heads, q_seq_len,
 * k_seq_len]. type = [bool].
 */
[[deprecated("use diopiAttention instead")]] DIOPI_API diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiTensorHandle_t q,
                                                                                            diopiTensorHandle_t k, diopiTensorHandle_t v, double dropout_p,
                                                                                            bool is_causal, bool return_debug_mask, double scale,
                                                                                            diopiTensorHandle_t out, diopiTensorHandle_t softmax_lse,
                                                                                            diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask);

// This interface has been deprecated, please use a more suitable diopiFlashAttention series interface.
/**
 * @brief Compute the forward pass for MultiheadAttention.
 * @param[in] ctx The diopi context.
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] q Query tensor from the forward pass. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor from the forward pass. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor from the forward pass. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] out Output tensor from the forward pass. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] softmax_lse Tensor representing the log-sum-exp of softmax values from the forward pass. shape = [batch_size, head_num, q_seq_len]. type =
 * [float32, float16, float64].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] gen Handle representing the random number generator used for dropout in the forward pass.
 * @param[in] scale Scaling factor used for attention weights in the forward pass. type = [float32, float16, float64].
 * @param[out] grad_q The gradient of the query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_k The gradient of the key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_v The gradient of the value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 */
[[deprecated("use diopiAttentionBackward instead")]] DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
    diopiConstTensorHandle_t out, diopiConstTensorHandle_t softmax_lse, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale,
    diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v);

// This interface has been deprecated, please use a more suitable diopiFlashAttention series interface.
/**
 * @brief Compute the forward pass for MultiheadAttentionVarLen.
 * @param[in] ctx The diopi context.
 * @param[in] q Query tensor. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 *        - For the implementation of flash-attention in CUDA, it is necessary to pad for 'q' to ensure that the 'head_dim' of the output from 'q' is
 * divisible by 8. Therefore, it is required to perform in-place modifications on 'q' by setting its data type to 'diopiTensorHandle_t'.
 * @param[in] k Key tensor. shape = [k_nums, head_num, head_dim]. type = [float32, float16, float64].
 *        - For the implementation of flash-attention in CUDA, it is necessary to pad for 'k' to ensure that the 'head_dim' of the output from 'k' is
 * divisible by 8. Therefore, it is required to perform in-place modifications on 'k' by setting its data type to 'diopiTensorHandle_t'.
 * @param[in] v Value tensor. shape = [v_nums, head_num, head_dim]. type = [float32, float16, float64].
 *        - For the implementation of flash-attention in CUDA, it is necessary to pad for 'v' to ensure that the 'head_dim' of the output from 'v' is
 * divisible by 8. Therefore, it is required to perform in-place modifications on 'v' by setting its data type to 'diopiTensorHandle_t'.
 * @param[in] cum_seq_q Cumulative sequence length for the query. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] cum_seq_k Cumulative sequence length for the key. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] max_q Maximum sequence length for the query. type = [int64, int32].
 * @param[in] max_k Maximum sequence length for the key. type = [int64, int32].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] return_debug_mask Flag indicating if the attention debug mask should be returned. type = [bool].
 * @param[in] scale Scaling factor for attention weights. type = [float32, float16, float64].
 * @param[out] out Tensor containing the result after applying multi-head attention. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] softmax_lse Tensor representing the log-sum-exp of the softmax values. shape = [batch_size, head_num, max_q]. type = [float32, float16, float64].
 * @param[out] gen Handle for the random number generator used in dropout.
 * @param[out] debug_attn_mask Debugging tensor for the attention mask (returned if return_debug_mask is true). shape = [batch_size, num_heads, max_q, max_k].
 * type = [bool].
 */
[[deprecated("use diopiAttentionVarLen instead")]] DIOPI_API diopiError_t diopiMultiHeadAttentionVarLen(
    diopiContextHandle_t ctx, diopiTensorHandle_t q, diopiTensorHandle_t k, diopiTensorHandle_t v, diopiConstTensorHandle_t cum_seq_q,
    diopiConstTensorHandle_t cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, bool return_debug_mask, double scale,
    diopiTensorHandle_t out, diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debug_attn_mask);

// This interface has been deprecated, please use a more suitable diopiFlashAttention series interface.
/**
 * @brief Compute the forward pass for MultiheadAttentionVarLen.
 * @param[in] ctx The diopi context.
 * @param[in] grad_out The gradient of the output tensor. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] q Query tensor from the forward pass. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor from the forward pass. shape = [k_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor from the forward pass. shape = [v_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] out Output tensor from the forward pass. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] softmax_lse Tensor representing the log-sum-exp of softmax values from the forward pass. shape = [batch_size, head_num, max_q]. type = [float32,
 * float16, float64].
 * @param[in] cum_seq_q Cumulative sequence length for the query. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] cum_seq_k Cumulative sequence length for the key. shape = [batch_size+1, ]. type = [int64, int32].
 * @param[in] max_q Maximum sequence length for the query. type = [int64, int32].
 * @param[in] max_k Maximum sequence length for the key. type = [int64, int32].
 * @param[in] dropout_p Dropout probability. type = [float32, float16, float64].
 * @param[in] is_causal Flag to determine if the attention should be causal, masking future tokens. type = [bool]
 * @param[in] gen Handle representing the random number generator used for dropout in the forward pass.
 * @param[in] scale Scaling factor used for attention weights in the forward pass. type = [float32, float16, float64].
 * @param[out] grad_q The gradient of the query tensor. shape = [q_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_k The gradient of the key tensor. shape = [k_nums, head_num, head_dim]. type = [float32, float16, float64].
 * @param[out] grad_v The gradient of the value tensor. shape = [v_nums, head_num, head_dim]. type = [float32, float16, float64].
 */
[[deprecated("use diopiAttentionVarLenBackward instead")]] DIOPI_API diopiError_t diopiMultiHeadAttentionVarLenBackward(
    diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
    diopiConstTensorHandle_t out, diopiConstTensorHandle_t softmax_lse, diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, int64_t max_q,
    int64_t max_k, double dropout_p, bool is_causal, diopiGeneratorHandle_t gen, double scale, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k,
    diopiTensorHandle_t grad_v);

/**
 * @brief Compute the forward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] gen Handle for the random number generator used in dropout op.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size, head_num, q_seq_len, head_dim] or [batch_size, q_seq_len,
 * hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim] or [batch_size, head_num, k_seq_len, head_dim] or [batch_size, k_seq_len,
 * hidden_size] or [k_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim] or [batch_size, head_num, v_seq_len, head_dim] or [batch_size, v_seq_len,
 * hidden_size] or [v_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] is_causal Whether to apply causal attention mask.
 * @param[in] head_num Number of heads. This parameter is required when the input tensor is 3D.
 * @param[in] input_layout The layout of input tensor. type = [char*], "BSND", "BNSD", "BSH", "SBH".
 * @param[out] attention_out Tensor storing the result after applying flash attention. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size,
 * head_num, q_seq_len, head_dim] or [batch_size, q_seq_len, hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[out] attention_mask Tensor storing the causal attention mask for back propagation. shape = [q_seq_len, k_seq_len]. type = [bool].
 * @param[out] dropout_mask Tensor storing the dropout mask for back propagation.
 * @param[out] softmax_max Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 * @param[out] softmax_sum Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 * @param[out] softmax_out Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 */
[[deprecated("use diopiAttention instead")]] DIOPI_API diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out,
                                                                                        diopiTensorHandle_t* attention_mask, diopiTensorHandle_t* dropout_mask,
                                                                                        diopiTensorHandle_t* softmax_max, diopiTensorHandle_t* softmax_sum,
                                                                                        diopiTensorHandle_t* softmax_out, diopiGeneratorHandle_t gen,
                                                                                        diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                                                                        diopiConstTensorHandle_t v, double p_dropout, double softmax_scale,
                                                                                        bool is_causal, int64_t head_num, const char* input_layout);

/**
 * @brief Compute the backward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size, head_num, q_seq_len, head_dim] or
 * [batch_size, q_seq_len, hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size, head_num, q_seq_len, head_dim] or [batch_size, q_seq_len,
 * hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim] or [batch_size, head_num, k_seq_len, head_dim] or [batch_size, k_seq_len,
 * hidden_size] or [k_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim] or [batch_size, head_num, v_seq_len, head_dim] or [batch_size, v_seq_len,
 * hidden_size] or [v_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] attention_out Tensor representing the forward calculation result. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size, head_num,
 * q_seq_len, head_dim] or [batch_size, q_seq_len, hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] attention_mask Tensor representing the causal attention mask from the forward pass. shape = [q_seq_len, k_seq_len]. type = [bool].
 * @param[in] dropout_mask Tensor representing the generated dropout mask from the forward pass.
 * @param[in] softmax_max Tensor representing the intermediate calculation result of softmax op from the forward pass. type = [float32].
 * @param[in] softmax_sum Tensor representing the intermediate calculation result of softmax op from the forward pass. type = [float32].
 * @param[in] softmax_out Tensor representing the intermediate calculation result of softmax op from the forward pass. type =[float32].
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] head_num Number of heads. This parameter is required when the input tensor is 3D.
 * @param[in] input_layout The layout of input tensor. type = [char*], "BSND", "BNSD", "BSH", "SBH".
 * @param[out] grad_q The gradient of the query tensor. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size, head_num, q_seq_len, head_dim] or
 * [batch_size, q_seq_len, hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[out] grad_k The gradient of the key tensor. shape = [batch_size, k_seq_len, head_num, head_dim] or [batch_size, head_num, k_seq_len, head_dim] or
 * [batch_size, k_seq_len, hidden_size] or [k_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[out] grad_v The gradient of the value tensor. shape = [batch_size, v_seq_len, head_num, head_dim] or [batch_size, head_num, v_seq_len, head_dim] or
 * [batch_size, v_seq_len, hidden_size] or [v_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 */
[[deprecated("use diopiAttentionBackward instead")]] DIOPI_API diopiError_t diopiFlashAttentionBackward(
    diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v, diopiConstTensorHandle_t grad_out,
    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t attention_out,
    diopiConstTensorHandle_t attention_mask, diopiConstTensorHandle_t dropout_mask, diopiConstTensorHandle_t softmax_max, diopiConstTensorHandle_t softmax_sum,
    diopiConstTensorHandle_t softmax_out, double p_dropout, double softmax_scale, int64_t head_num, const char* input_layout);

/**
 * @brief Computes attention on query, key and value tensors, using an optional attention mask if passed,
 and applying dropout if a probability greater than 0.0 is specified.
 * https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
 * @param[in] ctx The diopi context.
 * @param[out] attention_out Tensor storing the result after applying attention. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16,
 float16, float32].
 * @param[out] save_for_backward The intermediate variables that need to be saved for backward.How many intermediate variables need to be saved for forward
 calculation to be used in backward calculation varies in different devices and implementations. Specifies that the pre-allocated save_for_backward size cannot
 be less than 16
 * @param[out] save_tensor_num The number of the intermediate variables that need to be saved for backward
 * @param[in] attention_mask (optional Tensor) – Attention mask. shape (batch_size, q_seq_len, kv_seq_len) or (q_seq_len, kv_seq_len). A boolean mask
 where a value of True indicates that the element should take part in attention.
 * @param[in] attention_bias (optional Tensor) – Attention bias. shape = [(batch_size, head_num, q_seq_len, kv_seq_len), (head_num, q_seq_len, kv_seq_len),
 (q_seq_len, kv_seq_len)]. A float mask of the same type as query, key, value that is added to the attention score.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] k Key tensor. shape = [batch_size, kv_seq_len, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] v Value tensor. shape = [batch_size, kv_seq_len, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] p_dropout Dropout probability; if greater than 0.0, dropout is applied.
 * @param[in] gen_dropout Handle for the random number generator used in dropout op.
 * @param[in] softmax_scale The temperature to use for the softmax attention. if softmax_scale < 0, softmax_scale=\frac{1}{\sqrt{head_dim}}
 * @param[in] is_causal Whether to apply causal attention mask. If true, assumes causal attention masking and errors if both attn_mask and is_causal are set.
 */
DIOPI_API diopiError_t diopiAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t* save_for_backward,
                                      int64_t* save_tensor_num, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                      diopiConstTensorHandle_t attention_mask, diopiConstTensorHandle_t attention_bias, double p_dropout,
                                      diopiGeneratorHandle_t gen_dropout, double softmax_scale, bool is_causal);
/**
 * @brief Compute the backward pass for Attention.
 * https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
 * @param[in] ctx The diopi context.
 * @param[out] grad_q The gradient of the query tensor. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[out] grad_k The gradient of the key tensor. shape = [batch_size, kv_seq_len, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[out] grad_v The gradient of the value tensor. shape = [batch_size, kv_seq_len, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[out] grad_attn_bias (optional Tensor) The gradient of the atten_bias. shape = [(batch_size, q_seq_len, kv_seq_len), (q_seq_len, kv_seq_len)]. type =
 * [bfloat16, float16, float32].
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] k k Key tensor. shape = [batch_size, kv_seq_len, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] v Value tensor. shape = [batch_size, kv_seq_len, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] attention_out Tensor storing the result after attention. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16,
 * float16, float32].
 * @param[in] attention_mask (optional Tensor) – Attention mask. shape (batch_size, q_seq_len, kv_seq_len)
 * @param[in] save_for_backward The intermediate variables saved in forward
 * @param[out] save_tensor_num The number of the intermediate variables saved in forward
 * @param[in] p_dropout The probability of dropout op.
 * @param[in] softmax_scale The temperature to use for the softmax attention. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] gen_dropout Handle for the random number generator used in dropout op.
 */
DIOPI_API diopiError_t diopiAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v,
                                              diopiTensorHandle_t grad_attn_bias, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q,
                                              diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t attention_out,
                                              diopiConstTensorHandle_t* saved_for_backward, int64_t saved_tensor_num, double p_dropout,
                                              diopiGeneratorHandle_t gen_dropout, double softmax_scale);

/**
 * @brief Computes attention on query, key and value tensors, using an optional attention mask if passed,
 and applying dropout if a probability greater than 0.0 is specified.
 * https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
 * @param[in] ctx The diopi context.
 * @param[out] attention_out Tensor storing the result after applying attention. shape = [total_q, q_head_num, head_dim]. type = [bfloat16,
 float16, float32].
 * @param[out] save_for_backward The intermediate variables that need to be saved for backward.How many intermediate variables need to be saved for forward
 calculation to be used in backward calculation varies in different devices and implementations. Specifies that the pre-allocated save_for_backward size cannot
 be less than 16
 * @param[out] save_tensor_num The number of the intermediate variables that need to be saved for backward
 * @param[in] attention_mask (optional Tensor) – Attention mask. shape (total_q, total_kv). A boolean mask
 where a value of True indicates that the element should take part in attention.
 * @param[in] attention_bias (optional Tensor) – Attention bias. shape = [(total_q, total_kv)]. A float mask of the same type as query, key, value that is added
to the attention score.
 * @param[in] q Query tensor. shape = [total_q, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] k Key tensor. shape = [total_kv, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] v Value tensor. shape = [total_kv, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] cu_seqlens_q cu_seqlens_q tensor. shape = [batch_size + 1]. type = [int64_t]. cu_seqlens_q[:total_q] contains the position of the first token in
query for each batch. And cu_seqlens_q[batch_size] contains the total length of query. Note that cu_seqlens_q[i+1]-cu_seqlens_q[i] can calculate out the
sequence length of batch i.
 * @param[in] cu_seqlens_kv cu_seqlens_kv tensor. shape = [batch_size + 1]. type = [int64_t]. cu_seqlens_kv[:total_kv] contains the position of the first token
in key/value for each batch. And cu_seqlens_kv[batch_size] contains the total length of key/value. Note that cu_seqlens_kv[i+1]-cu_seqlens_kv[i] can calculate
out the sequence length of batch i.
 * @param[in] p_dropout Dropout probability; if greater than 0.0, dropout is applied.
 * @param[in] gen_dropout Handle for the random number generator used in dropout op.
 * @param[in] softmax_scale The temperature to use for the softmax attention. if softmax_scale < 0, softmax_scale=\frac{1}{\sqrt{head_dim}}
 * @param[in] is_causal Whether to apply causal attention mask. If true, assumes causal attention masking and errors if both attn_mask and is_causal are set.
 */
DIOPI_API diopiError_t diopiAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t* save_for_backward,
                                            int64_t* save_tensor_num, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                            diopiConstTensorHandle_t cu_seqlens_q, diopiConstTensorHandle_t cu_seqlens_kv, int64_t max_seqlen,
                                            int64_t max_kvlen, diopiConstTensorHandle_t attention_mask, diopiConstTensorHandle_t attention_bias,
                                            double p_dropout, diopiGeneratorHandle_t gen_dropout, double softmax_scale, bool is_causal);

/**
 * @brief Compute the backward pass for Attention.
 * https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
 * @param[in] ctx The diopi context.
 * @param[out] grad_q The gradient of the query tensor. shape = [total_q, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[out] grad_k The gradient of the key tensor. shape = [total_kv, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[out] grad_v The gradient of the value tensor. shape = [total_kv, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[out] grad_attn_bias (optional Tensor) The gradient of the atten_bias. shape = [(batch_size, q_seq_len, kv_seq_len), (q_seq_len, kv_seq_len)]. type =
 * [bfloat16, float16, float32].
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] q Query tensor. shape = [total_q, q_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] k k Key tensor. shape = [total_kv, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] v Value tensor. shape = [total_kv, kv_head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] cu_seqlens_q cu_seqlens_q tensor. shape = [batch_size + 1]. type = [int64_t]. cu_seqlens_q[:total_q] contains the position of the first token in
query for each batch. And cu_seqlens_q[batch_size] contains the total length of query. Note that cu_seqlens_q[i+1]-cu_seqlens_q[i] can calculate out the
sequence length of batch i.
 * @param[in] cu_seqlens_kv cu_seqlens_kv tensor. shape = [batch_size + 1]. type = [int64_t]. cu_seqlens_kv[:total_kv] contains the position of the first token
in key/value for each batch. And cu_seqlens_kv[batch_size] contains the total length of key/value. Note that cu_seqlens_kv[i+1]-cu_seqlens_kv[i] can calculate
out the sequence length of batch i.
 * @param[in] attention_out Tensor storing the result after applying flash attention. shape = [total_q, q_head_num, head_dim]. type = [bfloat16,
 * float16, float32].
 * @param[in] attention_mask (optional Tensor) – Attention mask. shape = [total_q, total_kv]
 * @param[in] save_for_backward The intermediate variables saved in forward
 * @param[out] save_tensor_num The number of the intermediate variables saved in forward
 * @param[in] p_dropout The probability of dropout op.
 * @param[in] softmax_scale The temperature to use for the softmax attention. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] gen_dropout Handle for the random number generator used in dropout op.
 */
DIOPI_API diopiError_t diopiAttentionVarLenBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k,
                                                    diopiTensorHandle_t grad_v, diopiTensorHandle_t grad_attn_bias, diopiConstTensorHandle_t grad_out,
                                                    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                    diopiConstTensorHandle_t cu_seqlens_q, diopiConstTensorHandle_t cu_seqlens_kv,
                                                    diopiConstTensorHandle_t attention_out, diopiConstTensorHandle_t* saved_for_backward,
                                                    int64_t saved_tensor_num, double p_dropout, diopiGeneratorHandle_t gen_dropout, double softmax_scale);

// The difference between this interface and the original diopiFlashAttention definition is that the passed input attention mask can be used directly. This
// prevents the attention mask from being recalculated inside the op. This helps reduce a lot of useless overhead when training large language models.
/**
 * @brief Compute the forward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] gen Handle for the random number generator used in dropout op.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size, head_num, q_seq_len, head_dim] or [batch_size, q_seq_len,
 * hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim] or [batch_size, head_num, k_seq_len, head_dim] or [batch_size, k_seq_len,
 * hidden_size] or [k_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim] or [batch_size, head_num, v_seq_len, head_dim] or [batch_size, v_seq_len,
 * hidden_size] or [v_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[in] attention_mask Casual attention mask tensor. When the tensor is empty, it means that the casual mask is not applied. shape = [q_seq_len,
 * k_seq_len]. type = [bool].
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] head_num Number of heads. This parameter is required when the input tensor is 3D.
 * @param[in] input_layout The layout of input tensor. type = [char*], "BSND", "BNSD", "BSH", "SBH".
 * @param[out] attention_out Tensor storing the result after applying flash attention. shape = [batch_size, q_seq_len, head_num, head_dim] or [batch_size,
 * head_num, q_seq_len, head_dim] or [batch_size, q_seq_len, hidden_size] or [q_seq_len, batch_size, hidden_size]. type = [bfloat16, float16].
 * @param[out] dropout_mask Tensor storing the dropout mask for back propagation.
 * @param[out] softmax_max Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 * @param[out] softmax_sum Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 * @param[out] softmax_out Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 */
[[deprecated("use diopiAttention instead")]] DIOPI_API diopiError_t diopiFlashAttentionV2(
    diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t* dropout_mask, diopiTensorHandle_t* softmax_max,
    diopiTensorHandle_t* softmax_sum, diopiTensorHandle_t* softmax_out, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
    diopiConstTensorHandle_t v, diopiConstTensorHandle_t attention_mask, double p_dropout, double softmax_scale, int64_t head_num, const char* input_layout);

// The difference between this interface and v1 v2 is that it is more suitable for devices integrating dropout mask and casual attention mask into the flash
// attention kernel.
/**
 * @brief Compute the forward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[inout] gen Handle for the random number generator used in dropout op.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] is_causal Whether to apply causal attention mask.
 * @param[out] softmax_lse Tensor storing the logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor). shape =
 [batch_size, head_num, q_seq_len]. type = [float32].
 * @param[out] attention_out Tensor storing the result after applying flash attention. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [bfloat16,
 * float16].
 */
[[deprecated("use diopiAttention instead")]] DIOPI_API diopiError_t diopiFlashAttentionV3(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out,
                                                                                          diopiTensorHandle_t softmax_lse, diopiGeneratorHandle_t gen,
                                                                                          diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                                                                          diopiConstTensorHandle_t v, double p_dropout, double softmax_scale,
                                                                                          bool is_causal);

/**
 * @brief Compute the backward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] gen Handle for the random number generator used in dropout op.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] attention_out Tensor representing the forward calculation result. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] softmax_lse Tensor representing the intermediate calculation result of softmax op from the forward pass.  shape = [batch_size, head_num,
 * q_seq_len]. type = [float32].
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] is_causal Whether to apply causal attention mask.
 * @param[out] grad_q The gradient of the query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[out] grad_k The gradient of the key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 * @param[out] grad_v The gradient of the value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [bfloat16, float16].
 */
[[deprecated("use diopiAttentionBackward instead")]] DIOPI_API diopiError_t diopiFlashAttentionV3Backward(
    diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v, diopiConstTensorHandle_t grad_out,
    diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t attention_out,
    diopiConstTensorHandle_t softmax_lse, double p_dropout, double softmax_scale, bool is_causal);

/**
 * @brief Compute the forward pass for the variable length version of Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[inout] gen Handle for the random number generator used in dropout op.
 * @param[in] q Query tensor. shape = [total_q, head_num, head_dim], where total_q = total number of query tokens in the batch. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [total_k, head_num, head_dim], where total_k = total number of key tokens in the batch. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [total_v, head_num, head_dim, where total_v = total number of value tokens in the batch. type = [bfloat16, float16].
 * @param[in] cum_seq_q The cumulative sequence lengths of the sequences in the batch for query. shape = [batch_size+1].
 * @param[in] cum_seq_kv The cumulative sequence lengths of the sequences in the batch for key and value. shape = [batch_size+1].
 * @param[in] max_seqlen_q Maximum sequence length for query.
 * @param[in] max_seqlen_kv Maximum sequence length for key and value.
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] is_causal Whether to apply causal attention mask.
 * @param[out] attention_out Tensor storing the result after applying flash attention. shape = [total, head_num, head_dim]. type = [bfloat16,
 * float16].
 * @param[out] attention_mask Tensor storing the causal attention mask for back propagation. shape = [total_q, total_k]. type = [bool].
 * @param[out] dropout_mask Tensor storing the dropout mask for back propagation.
 * @param[out] softmax_max Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 * @param[out] softmax_sum Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 * @param[out] softmax_out Tensor storing the intermediate calculation result of softmax op for back propagation. type = [float32].
 */
[[deprecated("use diopiAttentionVarLen instead")]] DIOPI_API diopiError_t
diopiFlashAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t* attention_mask, diopiTensorHandle_t* dropout_mask,
                          diopiTensorHandle_t* softmax_max, diopiTensorHandle_t* softmax_sum, diopiTensorHandle_t* softmax_out, diopiGeneratorHandle_t gen,
                          diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiSize_t cum_seq_q, diopiSize_t cum_seq_kv,
                          int64_t max_seqlen_q, int64_t max_seqlen_kv, double p_dropout, double softmax_scale, bool is_causal);

/**
 * @brief Compute the backward pass for the variable length version of Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] grad_out The gradient of the output tensor. shape = [total, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] q Query tensor. shape = [total_q, head_num, head_dim], where total_q = total number of query tokens in the batch. type = [bfloat16, float16].
 * @param[in] k Key tensor. shape = [total_k, head_num, head_dim], where total_k = total number of key tokens in the batch. type = [bfloat16, float16].
 * @param[in] v Value tensor. shape = [total_v, head_num, head_dim, where total_v = total number of value tokens in the batch. type = [bfloat16, float16].
 * @param[in] cum_seq_q The cumulative sequence lengths of the sequences in the batch for query. shape = [batch_size+1].
 * @param[in] cum_seq_kv The cumulative sequence lengths of the sequences in the batch for key and value. shape = [batch_size+1].
 * @param[in] attention_out Tensor representing the forward calculation result. shape = [total, head_num, head_dim]. type = [bfloat16, float16].
 * @param[in] attention_mask Tensor representing the causal attention mask from the forward pass. shape = [max_seqlen_q, max_seqlen_kv]. type = [bool].
 * @param[in] dropout_mask Tensor representing the generated dropout mask from the forward pass.
 * @param[in] softmax_max Tensor representing the intermediate calculation result of softmax op from the forward pass. type = [float32].
 * @param[in] softmax_sum Tensor representing the intermediate calculation result of softmax op from the forward pass. type = [float32].
 * @param[in] softmax_out Tensor representing the intermediate calculation result of softmax op from the forward pass. type =[float32].
 * @param[in] max_seqlen_q Maximum sequence length for query.
 * @param[in] max_seqlen_kv Maximum sequence length for key and value.
 * @param[in] p_dropout Dropout probability.
 * @param[in] softmax_scale The scaling of qk^T before applying softmax. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[out] grad_q The gradient of the query tensor. shape = [total_q, head_num, head_dim], where total_q = total number of query tokens in the batch. type =
 * [bfloat16, float16].
 * @param[out] grad_k The gradient of the key tensor. shape = [total_k, head_num, head_dim], where total_k = total number of key tokens in the batch. type =
 * [bfloat16, float16].
 * @param[out] grad_v The gradient of the value tensor. shape = [total_v, head_num, head_dim], where total_v = total number of value tokens in the batch. type =
 * [bfloat16, float16].
 */
[[deprecated("use diopiAttentionVarLenBackward instead")]] DIOPI_API diopiError_t diopiFlashAttentionVarLenBackward(
    diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v, diopiConstTensorHandle_t grad_out,
    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiSize_t cum_seq_q, diopiSize_t cum_seq_kv,
    diopiConstTensorHandle_t attention_out, diopiConstTensorHandle_t attention_mask, diopiConstTensorHandle_t dropout_mask,
    diopiConstTensorHandle_t softmax_max, diopiConstTensorHandle_t softmax_sum, diopiConstTensorHandle_t softmax_out, int64_t max_seqlen_q,
    int64_t max_seqlen_kv, double p_dropout, double softmax_scale);

// This interface is temporarily designed for ascend, please do not use it with other devices.
/**
 * @brief Compute the forward pass for scaled masked softmax op.
 * @param[in] ctx The diopi context.
 * @param[in] input Input tensor. shape = [batch_size, seq_len, head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] mask Mask tensor. shape = [batch_size, seq_len, head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] scale Scale for softmax op.
 * @param[in] fixed_triu_mask Whether to use a fixed triu mask.
 * @param[out] output Output tensor storing the calculation result of scaled masked softmax op. type = [bfloat16, float16, float32].
 */
DIOPI_API diopiError_t diopiScaledMaskedSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiConstTensorHandle_t input,
                                                diopiConstTensorHandle_t mask, double scale, bool fixed_triu_mask);

// This interface is temporarily designed for ascend, please do not use it with other devices.
/**
 * @brief Compute the backward pass for scaled masked softmax op.
 * @param[in] ctx The diopi context.
 * @param[in] grad_output The gradient of output tensor. shape = [batch_size, seq_len, head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] output Tensor representing the result of scaled masked softmax op. shape = [batch_size, seq_len, head_num, head_dim]. type =
 * [bfloat16, float16, float32].
 * @param[in] mask Mask tensor. shape = [batch_size, seq_len, head_num, head_dim]. type = [bfloat16, float16, float32].
 * @param[in] scale Scale for softmax op.
 * @param[in] fixed_triu_mask Whether to use a fixed triu mask.
 * @param[out] grad_input The gradient of input tensor. type = [bfloat16, float16, float32].
 */
DIOPI_API diopiError_t diopiScaledMaskedSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                        diopiConstTensorHandle_t output, diopiConstTensorHandle_t mask, double scale, bool fixed_triu_mask);

// ============================================lightllm begin========================================

/**
 * @brief This function applies a penalty to the given logits based on the presence and frequency of certain tokens in the input sequence to suppress
 * generating tokens repeatedly.
 * The p_cumsum_seq_len is used to determine the sequence length, which is then used to extract the corresponding token_id from p_token_ids and
 * token_count from p_token_counts.
 * For each token，the final logit = logit - corresponding_presence_penalty * token_counts - corresponding_presence_penalty.
 * @param[in] ctx The diopi context.
 * @param[inout] logits Tensor representing the logits. Shape: [batch_size, voc_len]. It contains the predicted scores for each token in the input sequences.
 * It will be penalized by frequency_penalty and presence_penalty.
 * @param[in] presence_penalty Tensor representing the presence penalty for each batch. Shape: [batch_size,]. It contains the penalty values to be subtracted
 * from the logits.
 * @param[in] frequency_penalty Tensor representing the frequency penalty for each batch. Shape: [batch_size,]. It contains the penalty values to be subtracted
 * from the logits.
 * @param[in] p_token_ids Tensor representing the token_ids for generated tokens. Shape:[generated_tokens_num].
 * @param[in] p_token_counts Tensor representing the count of each token for generated tokens. Shape:[generated_tokens_num].
 * @param[in] p_cumsum_seq_len Tensor representing the cumulative sum of sequece lengths of each batch. Shape:[batch_size+1].
 * It contains the indices indicating the satrt and end positions of each batch in the p_token_ids and p_token_counts tensors.
 * @param[in] p_max_len_in_batch: Scalar representing the maximum length among the sequeces in a batch.
 */
DIOPI_API diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presence_penalty,
                                         diopiConstTensorHandle_t frequency_penalty, diopiConstTensorHandle_t p_token_ids,
                                         diopiConstTensorHandle_t p_token_counts, diopiConstTensorHandle_t p_cumsum_seq_len, int p_max_len_in_batch);

/**
 * @brief Copies the elements from k tensor into out tensor according to dest_loc tensor. It can be expressed in detail as: out[dest_loc] = k. During
 * model initialization, the KV cache is pre-allocated based on the user-set max_total_token_num and a Token Table is created to record the actual storage
 * locations of input tokens. For details, please refer to the official implementation using the triton kernel:
 * https://github.com/ModelTC/lightllm/blob/main/docs/TokenAttention.md.
 * https://github.com/ModelTC/lightllm/blob/main/lightllm/common/basemodel/triton_kernel/destindex_copy_kv.py.
 * @param[in] ctx diopi context.
 * @param[in] k Tensor representing the src tensor to be copied. shape = [seq_len, head_num, head_dim].
 * @param[in] dest_loc Tensor representing the destination location to be covered by the src tensor in the out tensor. shape = [seq_len, ].
 * @param[out] out Tensor representing the output tensor that needs to be partially covered by the src tensor based on the destination location tensor. shape =
 * [max_total_token_num, head_num, head_dim].
 */
DIOPI_API diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t dest_loc);

/**
 * @brief The no pad implementation of \text{token_attention_out}(\mathrm{q},\mathrm{k})=\frac{\mathrm{qk}^\mathrm{T}}{\sqrt{\mathrm{d_k}}}.
 * For details, please refer to the official implementation using the triton kernel:
 * https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py.
 * @param[in] ctx diopi context.
 * @param[in] q Tensor representing the query matrix in the attention mechanism. shape = [batch_size, head_num, head_dim].
 * @param[in] k Tensor representing the key matrix in the attention mechanism. shape = [max_total_token_num, head_num, head_dim].
 * @param[in] b_loc Tensor representing the locations of all tokens in the sequence in each batch. shape = [batch_size, N].
 * @param[in] b_start_loc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size].
 * @param[in] b_seq_len Tensor representing the sequence length in each batch. shape = [batch_size].
 * @param[in] max_input_len The maximum length of all batch corresponding sequences.
 * @param[out] token_attention_out The output tensor of token attention's calculation. shape = [head_num, sum_batch_seq_len]. sum_batch_seq_len is the sum of
 * the lengths of all batch corresponding sequences, and also the sum of the elements in b_seq_len tensor.
 */
DIOPI_API diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t token_attention_out, diopiConstTensorHandle_t q,
                                                    diopiConstTensorHandle_t k, diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc,
                                                    diopiConstTensorHandle_t b_seq_len, int max_input_len);

/**
 * @brief The nopad implementation of \mathrm{out}=\mathrm{softmax(\mathrm{logics})}*\mathrm{v}. For details, please refer to the official implementation using
 * the triton kernel: https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_softmax_and_reducev.py.
 * @param[in] ctx The diopi context.
 * @param[in] logics Tensor representing the input tensor. shape = [head_num, sum_batch_seq_len]. sum_batch_seq_len is the sum of the
 * lengths of all batch corresponding sequences, and also the sum of the elements in b_seq_len tensor.
 * @param[in] v Tensor representing the value matrix in the attention mechanism. shape = [max_total_token_num, head_num, head_dim].
 * @param[in] b_loc Tensor representing the locations of all tokens in the sequence in each batch. shape = [batch_size, N].
 * @param[in] b_start_loc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size].
 * @param[in] b_seq_len Tensor representing the sequence length in each batch. shape = [batch_size].
 * @param[in] max_input_len The maximum length of all batch corresponding sequences.
 * @param[in] other_kv_index To avoid reading nan data, other_kv_index is set as b_loc[0, max_input_len - 1].item().
 * @param[in] out The output tensor of softmax_reduceV operation. shape = [batch_size, head_num, head_dim].
 */
DIOPI_API diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics,
                                                         diopiConstTensorHandle_t v, diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc,
                                                         diopiConstTensorHandle_t b_seq_len, int max_input_len, int other_kv_index);

/**
 * @brief The no pad implementation of
 * \text{context_attention_out}(\mathrm{q},\mathrm{k},\mathrm{v})=\text{softmax}(\frac{\mathrm{qk}^\mathrm{T}}{\sqrt{\mathrm{d_k}}})\mathrm{v}. For details,
 * please refer to the official implementation using the triton kernel:
 * https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py.
 * @param[in] ctx diopi context.
 * @param[in] q Tensor representing the query matrix in the attention mechanism. shape = [sum_batch_seq_len, head_num, head_dim]. sum_batch_seq_len is the sum
 * of the lengths of all batch corresponding sequences, and also the sum of the elements in b_seq_len tensor.
 * @param[in] k Tensor representing the key matrix in the attention mechanism. shape = [sum_batch_seq_len, head_num, head_dim]
 * @param[in] v Tensor representing the value matrix in the attention mechanism. shape = [sum_batch_seq_len, head_num, head_dim]
 * @param[in] b_start_loc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size]
 * @param[in] b_seq_len Tensor representing the sequence length in each batch. shape = [batch_size]
 * @param[in] max_input_len The maximum length of all batch corresponding sequences.
 * @param[in] context_attention_out The output tensor of context attention operation. shape = [sum_batch_seq_len, head_num, head_dim]
 */
DIOPI_API diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t context_attention_out, diopiConstTensorHandle_t q,
                                                      diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t b_start_loc,
                                                      diopiConstTensorHandle_t b_seq_len, int max_input_len);

// ============================================lightllm end========================================

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_
