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
 * @brief Copies the elements from K tensor into Out tensor according to DestLoc tensor. It can be expressed in detail as: Out[:DestLoc.shape[0]] = K. During
 * model initialization, the KV cache is pre-allocated based on the user-set max_total_token_num and a Token Table is created to record the actual storage
 * locations of input tokens. For details, please refer to the original source: https://github.com/ModelTC/lightllm/blob/main/docs/TokenAttention.md.
 * @param[in] ctx diopi context.
 * @param[in] K Tensor representing the src tensor to be copied. shape = [seq_len, head_num, head_dim].
 * @param[in] DestLoc Tensor representing the destination location to be covered by the src tensor in the out tensor. DestLoc = torch.arange(0, seq_len,
 * dtype=torch.int32). shape = [seq_len, ].
 * @param[out] Out Tensor representing the output tensor that needs to be partially covered by the src tensor based on the destination location tensor. shape =
 * [max_total_token_num, head_num, head_dim].
 */
DIOPI_API diopiError_t diopiDestindexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t K, diopiConstTensorHandle_t DestLoc);

/**
 * @brief \mathrm{AttentionOut}(\mathrm{Q},\mathrm{K})=\frac{\mathrm{QK}^\mathrm{T}}{\sqrt{\mathrm{d_k}}} For details, please refer to the original source:
 * https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py.
 * @param[in] ctx diopi context.
 * @param[in] Q Tensor representing the query matrix in the self-attention mechanism. shape = [batch_size, head_num, head_dim]
 * @param[in] K Tensor representing the key matrix in the self-attention mechanism. shape = [max_total_token_num, head_num, head_dim]
 * @param[in] BLoc Tensor representing the locations of all tokens in the sequence in each batch. shape = [batch_size, maxInputLen]
 * @param[in] BStartLoc Tensor representing the starting location of each batch in the entire sequence. shape = [batch_size, ]
 * @param[in] BSeqlen Tensor representing the sequence length in each batch. shape = [batch_size, ]
 * @param[in] maxInputLen The maximum sequence length in all batches.
 * @param[out] AttentionOut The output tensor of token attention's calculation. shape = [head_num, batch_size * maxInputLen]
 */
DIOPI_API diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t AttentionOut, diopiConstTensorHandle_t Q,
                                                    diopiConstTensorHandle_t K, diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc,
                                                    diopiConstTensorHandle_t BSeqlen, int maxInputLen);

/**
 * @brief Applies a softmax function. \mathrm{ProbOut}=\mathrm{softmax}(\mathrm{Logics})
 * @param[in] ctx diopi context.
 * @param[in] Logics
 * @param[in] BStartLoc
 * @param[in] BSeqlen
 * @param[in] maxInputLen
 * @param[in] ProbOut
 */
DIOPI_API diopiError_t diopiTokenSoftmaxInference(diopiContextHandle_t ctx, diopiTensorHandle_t ProbOut, diopiConstTensorHandle_t Logics,
                                                  diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen, int maxInputLen);

/**
 * @brief \mathrm{Out}=\mathrm{Prob}*\mathrm{V}
 * @param[in] ctx diopi context.
 * @param[in] Prob
 * @param[in] V
 * @param[in] BLoc
 * @param[in] BStartLoc
 * @param[in] BSeqlen
 * @param[in] maxInputLen
 * @param[in] Out
 */
DIOPI_API diopiError_t diopiTokenReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t Prob, diopiConstTensorHandle_t V,
                                                  diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc, diopiConstTensorHandle_t BSeqlen,
                                                  int maxInputLen);

/**
 * @brief \mathrm{Out}=\mathrm{softmax(\mathrm{Logics})}*\mathrm{V}
 * @param[in] ctx diopi context.
 * @param[in] Logics
 * @param[in] V
 * @param[in] BLoc
 * @param[in] BStartLoc
 * @param[in] BSeqlen
 * @param[in] maxInputLen
 * @param[in] otherKVIndex
 * @param[in] Out
 */
DIOPI_API diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t Out, diopiConstTensorHandle_t Logics,
                                                         diopiConstTensorHandle_t V, diopiConstTensorHandle_t BLoc, diopiConstTensorHandle_t BStartLoc,
                                                         diopiConstTensorHandle_t BSeqlen, int maxInputLen, diopiConstTensorHandle_t otherKVIndex);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LIGHTLLM_H_
