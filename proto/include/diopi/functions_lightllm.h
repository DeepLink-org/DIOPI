/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LIGHTLLM_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LIGHTLLM_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/**
 * @brief Copies the elements from K tensor into Out tensor according to DestLoc tensor. It can be expressed in detail as: Out[DestLoc] = K. During
 * model initialization, the KV cache is pre-allocated based on the user-set max_total_token_num and a Token Table is created to record the actual storage
 * locations of input tokens. For details, please refer to the official implementation using the triton kernel:
 * https://github.com/ModelTC/lightllm/blob/main/docs/TokenAttention.md.
 * https://github.com/ModelTC/lightllm/blob/main/lightllm/common/basemodel/triton_kernel/destindex_copy_kv.py.
 * @param[in] ctx diopi context.
 * @param[in] K Tensor representing the src tensor to be copied. shape = [seq_len, head_num, head_dim].
 * @param[in] DestLoc Tensor representing the destination location to be covered by the src tensor in the out tensor. shape = [seq_len, ].
 * @param[out] Out Tensor representing the output tensor that needs to be partially covered by the src tensor based on the destination location tensor. shape =
 * [max_total_token_num, head_num, head_dim].
 */
DIOPI_API diopiError_t diopiDestindexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t K, diopiConstTensorHandle_t DestLoc);

/**
 * @brief The nopad implementation of \mathrm{AttentionOut}(\mathrm{Q},\mathrm{K})=\frac{\mathrm{QK}^\mathrm{T}}{\sqrt{\mathrm{d_k}}}.
 * For details, please refer to the official implementation using the triton kernel:
 * https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py.
 * @param[in] ctx diopi context.
 * @param[in] Q Tensor representing the query matrix in the attention mechanism. shape = [batch_size, head_num, head_dim].
 * @param[in] K Tensor representing the key matrix in the attention mechanism. shape = [max_total_token_num, head_num, head_dim].
 * @param[in] BLoc Tensor representing the locations of all tokens in the sequence in each batch. shape = [batch_size, N].
 * @param[in] BStartLoc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size, ].
 * @param[in] BSeqlen Tensor representing the sequence length in each batch. shape = [batch_size, ].
 * @param[in] maxInputLen The maximum length of all batch corresponding sequences.
 * @param[out] AttentionOut The output tensor of token attention's calculation. shape = [head_num, sum_batch_seq_len]. sum_batch_seq_len is the sum of the
 * lengths of all batch corresponding sequences, and also the sum of the elements in BSeqlen tensor.
 */
DIOPI_API diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t AttentionOut, diopiConstTensorHandle_t Q,
                                                    diopiConstTensorHandle_t K, diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc,
                                                    diopiConstTensorHandle_t BSeqlen, int maxInputLen);

/**
 * @brief The nopad implementation of \mathrm{Out}=\mathrm{softmax(\mathrm{Logics})}*\mathrm{V}. For details, please refer to the official implementation using
 * the triton kernel: https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_softmax_and_reducev.py.
 * @param[in] ctx diopi context.
 * @param[in] Logics Tensor representing the value matrix in the attention mechanism. shape = [head_num, sum_batch_seq_len]. sum_batch_seq_len is the sum of the
 * lengths of all batch corresponding sequences, and also the sum of the elements in BSeqlen tensor.
 * @param[in] V Tensor representing the value matrix in the attention mechanism. shape = [max_total_token_num, head_num, head_dim].
 * @param[in] BLoc Tensor representing the locations of all tokens in the sequence in each batch. shape = [batch_size, N].
 * @param[in] BStartLoc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size, ].
 * @param[in] BSeqlen Tensor representing the sequence length in each batch. shape = [batch_size, ].
 * @param[in] maxInputLen The maximum length of all batch corresponding sequences.
 * @param[in] otherKVIndex To avoid reading nan data, other_kv_index is set as BLoc[0, maxInputLen - 1].item().
 * @param[in] Out The output tensor of softmax_reduceV operation. shape = [batch_size, head_num, head_dim].
 */
DIOPI_API diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t Logics,
                                                         diopiConstTensorHandle_t V, diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc,
                                                         diopiConstTensorHandle_t BSeqlen, int maxInputLen, int otherKVIndex);

/**
 * @brief The nopad implementation of \mathrm{Out}=\mathrm{softmax(\mathrm{Logics})}*\mathrm{V}. For details, please refer to the official implementation using
 * the triton kernel: https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py.
 * @param[in] ctx diopi context.
 * @param[in] Q Tensor representing the query matrix in the attention mechanism. shape = [sum_batch_seq_len, head_num, head_dim]. sum_batch_seq_len is the sum
 * of the lengths of all batch corresponding sequences, and also the sum of the elements in BSeqlen tensor.
 * @param[in] K Tensor representing the key matrix in the attention mechanism. shape = [sum_batch_seq_len, head_num, head_dim]
 * @param[in] V Tensor representing the value matrix in the attention mechanism. shape = [sum_batch_seq_len, head_num, head_dim]
 * @param[in] BStartLoc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size, ]
 * @param[in] BSeqlen Tensor representing the sequence length in each batch. shape = [batch_size, ]
 * @param[in] maxInputLen The maximum length of all batch corresponding sequences.
 * @param[in] Out The output tensor of softmax_reduceV operation. shape = [sum_batch_seq_len, head_num, head_dim]
 */
DIOPI_API diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t Q, diopiConstTensorHandle_t K,
                                                      diopiConstTensorHandle_t V, diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen,
                                                      int maxInputLen);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LIGHTLLM_H_
