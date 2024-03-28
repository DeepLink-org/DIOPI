/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LMDEPLOY_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LMDEPLOY_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/**
 * @brief Fused Silu FFN layer.(Silu(x * W1) dot (x * W3)) * W2.
 * @param[in] ctx diopi context.
 * @param[inout] inoutput : Output tensor.shape = [token_num, hidden_units].type = [float32, float16]
 * @param[in] weight1 : Weight1.shape = [hidden_units, inter_size].type = [float32, float16]
 * @param[in] weight2 : Weight2.shape = [hidden_units, inter_size].type = [float32, float16]
 * @param[in] weight3 : Weight3.shape = [inter_size, hidden_units].type = [float32, float16]
 * @param[inout] workspace : Workspace or buffer.type = [float32, float16]
 * @param[inout] workspace_size : Workspace size, if workspace_size < 0 then only cal workspace_size.type = [int64*, int32*]
 * @param[in] fusion_level : Fusion level, 0 represents no fusion, and the higher the numerical value, the higher the degree of fusion.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiFusedSiluFfnInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t weight1,
                                            diopiConstTensorHandle_t weight2, diopiConstTensorHandle_t weight3, diopiTensorHandle_t workspace,
                                            int64_t* workspace_size, int64_t fusion_level);

/**
 * @brief Root Mean Square (RMS) Normalization to the input tensor. without bias in interlm.
 * @param[in] ctx The diopi context.
 * @param[inout] inoutput : Inoutput tensor.shape = [num_token, hidden_units].type = [float32, float16]
 * @param[in] scale : The gain parameter used to re-scale the standardized summed inputs.shape = [hidden_units].type = [float32, float16]
 * @param[in] eps : A small value to avoid division by zero.type = [float32]
 */
DIOPI_API diopiError_t diopiRootMeanSquareNormInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t scale, float eps);

/**
 * @brief Fused res and bias and RMSNorm. 1.(res + inout + bias) = new_res 2. RMSNorm(new_res) = inout.
 * @param[in] ctx The diopi context.
 * @param[inout] inoutput : Inoutput tensor.shape = [num_token, hidden_units].type = [float32, float16]
 * @param[inout] residual : Residual tensor.shape = [num_token, hidden_units].type = [float32, float16]
 * @param[in] bias : The bias tensor.shape = [hidden_units].type = [float32, float16]
 * @param[in] scale : The gain parameter used to re-scale the standardized summed inputs.shape = [hidden_units].type = [float32, float16]
 * @param[in] eps : A small value to avoid division by zero.type = [float32]
 */
DIOPI_API diopiError_t diopiFusedAddRootMeanSquareNormInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiTensorHandle_t residual,
                                                          diopiConstTensorHandle_t bias, diopiConstTensorHandle_t scale, float eps);

/**
 * @brief FusedContextAttention.
 * 1.If pre_work like attention_mask and padding_offset and cu_seqlens is needed, get prepared result.Or only do the pre work.Or get size only.
 * attention_mask : Attention mask.shape = [batch_size, 1, max_query_len, max_key_len].type = [float32, float16]
 * padding_offset : Padding offset.shape = [token_num].type = [int64, int32]
 * cu_seqlens : Cuseqlens.shape = [batch_size + 1].type = [int64, int32]
 * 2.get QKV.
 * 3.qkv add bias & apply rotary embedding with his & rebuild padding & transpose.qkv [B, max_q_len, H + 2kvH, D] -> (q [B, H, max_q_len, D], k [B, kvH,
 * max_q_len, D], v [B, kvH, max_q_len, D])
 * 4.insert the k/v computed from inputs into k/v cache, k/v [B, kvH, s, D] -> k/v_cache [B, kvH, S[t:t+s], D].For
 * faster, [B, kvH, S[t:t+s], D/x, x] like is also ok, but shape mast same in decoder attn.
 * 5.copy kv from k/v cache back with extend for GQA/MQA.[B, kvH, S[:t+s], D/x, x] -> [B, qH, t+s, D]
 * 6.softmax(QK)*V batch gemm -> [B, H, S, D]. In softmax eps = 1e-6f and qk_val = qk_scale * qk_val - qk_bias with qu_bias
 * if masked  -10000.0f else 0, and with qk_scale = static_cast<T>(1.f / sqrtf(size_per_head_ * 1.f).
 * 7.transpose back and move padding -> [token_num,
 * hidden_units] 8.use_logn_attn use_dynamic_ntk will append future attention_mask: create a [batch_size, 1, max_query_len, max_key_len] tensor, while is_valid
 * = q < input_length && k < context_lengths && k <= q + (context_lengths - input_length) cu_seqlens: accumulated input_lengths, from 0 to all, with batch_size
 * + 1 numbers. padding_offset: cal padding offset, which means the total pad number before the token, when every seq is padded to same length with filling in
 * spaces after input.
 * @param[in] ctx diopi context.
 * @param[inout] inoutput : Inoutput tensor.shape = [token_num, hidden_units].type = [float32, float16]
 * @param[in] qkv_weight : QKV_weight tensor.shape = [hidden_units, (local_head_num+local_kv_head_num*2)*size_per_head].type = [float32, float16]
 * @param[in] qkv_bias : QKV_bias tensor.shape = [1, (local_head_num+local_kv_head_num*2)*size_per_head].type = [float32, float16]
 * @param[inout] pre_work : Pre work like attention_mask and padding_offset and cu_seqlens for lmdeploy.shape = [pre_work_size].type = [float32, float16]
 * @param[inout] pre_work_size : Pre workspace size, if pre_work_size < 0 then only cal pre_work_size.type = [int64*, int32*]
 * @param[in] is_prepared : All pre work is prepared or not.type = [bool]
 * @param[inout] workspace : Workspace or buffer.shape = [workspace_size].type = [float32, float16]
 * @param[inout] workspace_size : Workspace size, if workspace_size < 0 then only cal workspace_size.type = [int64*, int32*]
 * @param[in] fusion_level : Fusion level, 0 represents no fusion, and the higher the numerical value, the higher the degree of fusion.type = [int64, int32]
 * @param[inout] key_cache : Key cache.shape = [num_layer, batch, local_head_num, max_seq_len, size_per_head].type = [float32, float16]
 * @param[inout] value_cache : Value cache.shape = [num_layer, batch, local_head_num, max_seq_len, size_per_head].type = [float32, float16]
 * @param[in] input_lengths : Input lengths.shape = [batch_size].type = [int64, int32]
 * @param[in] history_lengths : History lengths.shape = [batch_size].type = [int64, int32]
 * @param[in] context_lengths : Contextlengths.shape = [batch_size].type = [int64, int32]
 * @param[in] layer_id : Layer id.type = [int64, int32]
 * @param[in] local_head_num : Local_head_num of q.type = [int64, int32]
 * @param[in] local_kv_head_num : Local_kv_head_num of kv.type = [int64, int32]
 * @param[in] size_per_head : Size per head.type = [int64, int32]
 * @param[in] max_seq_len : Max length of seq.type = [int64, int32]
 * @param[in] max_q_len : Max length of Q.type = [int64, int32]
 * @param[in] max_kv_len : Max length of KV.type = [int64, int32]
 * @param[in] rotary_embedding : Rotary_embedding.type = [int64, int32]
 * @param[in] rope_theta : Rotary_base.type = [float32]
 */
DIOPI_API diopiError_t diopiFusedContextAttentionInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t qkv_weight,
                                                     diopiConstTensorHandle_t qkv_bias, diopiTensorHandle_t pre_work, int64_t* pre_work_size, bool is_prepared,
                                                     diopiTensorHandle_t workspace, int64_t* workspace_size, int64_t fusion_level,
                                                     diopiTensorHandle_t key_cache, diopiTensorHandle_t value_cache, int64_t layer_id, int64_t local_head_num,
                                                     int64_t local_kv_head_num, int64_t size_per_head, int64_t max_seq_len, int64_t max_q_len,
                                                     int64_t max_kv_len, int64_t rotary_embedding, float rope_theta);

/**
 * @brief FusedDecoderAttention.
 * 1.get QKV.
 * 2.qkv add bias & apply rotary embedding, tstep = step - 1 - padd_len
 * 3.With GQA/MQA/MHA support, softmax(QK)*V.qk *= inv_sqrt_dh == 1.F / (sqrtf((float)size_per_head)) and eps = 1.e-6f.
 * 4.Kv saved to cache in 3, and if seq > max_seq, cache index = seq%max_seq
 * @param[in] ctx diopi context.
 * @param[inout] inoutput : Inoutput tensor.shape = [batch_size, hidden_units].type = [float32, float16]
 * @param[in] qkv_weight : QKV_weight tensor.shape = [hidden_units, (local_head_num+local_kv_head_num*2)*size_per_head].type = [float32, float16]
 * @param[in] qkv_bias : QKV_bias tensor.shape = [1, (local_head_num+local_kv_head_num*2)*size_per_head].type = [float32, float16]
 * @param[inout] workspace : Workspace or buffer.shape = [workspace_size].type = [float32, float16]
 * @param[inout] workspace_size : Workspace size, if workspace_size < 0 then only cal workspace_size.type = [int64*, int32*]
 * @param[in] fusion_level : Fusion level, 0 represents no fusion, and the higher the numerical value, the higher the degree of fusion.type = [int64, int32]
 * @param[inout] key_cache : Key cache.shape = [num_layer, batch, local_head_num, max_seq_len, size_per_head].type = [float32, float16]
 * @param[inout] value_cache : Value cache.shape = [num_layer, batch, local_head_num, max_seq_len, size_per_head].type = [float32, float16]
 * @param[in] finished : Finished batch.shape = [batch_size].type = [bool]
 * @param[in] total_padding_tokens : Total padding tokens.shape = [batch_size].type = [int64, int32]
 * @param[in] sequence_lengths : Sequence lengths.shape = [batch_size].type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 * @param[in] layer_id : Layer id.type = [int64, int32]
 * @param[in] local_head_num : Local_head_num of q.type = [int64, int32]
 * @param[in] local_kv_head_num : Local_kv_head_num of kv.type = [int64, int32]
 * @param[in] size_per_head : Size per head.type = [int64, int32]
 * @param[in] max_seq_len : Max length of seq.type = [int64, int32]
 * @param[in] rotary_embedding : Rotary_embedding.type = [int64, int32]
 * @param[in] rope_theta : Rotary_base.type = [float32]
 */
DIOPI_API diopiError_t diopiFusedDecoderAttentionInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t qkv_weight,
                                                     diopiConstTensorHandle_t qkv_bias, diopiTensorHandle_t workspace, int64_t* workspace_size,
                                                     int64_t fusion_level, diopiTensorHandle_t key_cache, diopiTensorHandle_t value_cache,
                                                     diopiConstTensorHandle_t finished, diopiConstTensorHandle_t total_padding_tokens,
                                                     diopiConstTensorHandle_t sequence_lengths, int64_t step, int64_t layer_id, int64_t local_head_num,
                                                     int64_t local_kv_head_num, int64_t size_per_head, int64_t max_seq_len, int64_t rotary_embedding,
                                                     float rope_theta);

/**
 * @brief SetupTopkRuntimeArgs. Fix topk and topp for each batch.
 * 1.if top_ks/ps_size > 0 k/p get from top_ks/ps else top_k/p
 * 2.if k == 0 && p == 0.0f then k = 1
 * 3.if k > 0 && p == 0.0f then p = 1.0f
 * 4.Clip k to <=1024, clip p to [0.0, 1.0] and save
 * 5.if k == 0 means skip needed then save to skip_decode
 * @param[in] ctx The diopi context.
 * @param[inout] top_ks : TopKs tensor.shape = [batch_size].type = [int64, int32]
 * @param[inout] top_ps : TopPs tensor.shape = [batch_size].type = [float32]
 * @param[inout] skip_decode : Need skip or not.shape = [batch_size].type = [bool]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] top_k : Topk.type = [int64, int32]
 * @param[in] top_ks_size : Topks size.type = [int64, int32]
 * @param[in] top_p : Topp.type = [float32]
 * @param[in] top_ps_size : Topps size.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiSetupTopkRuntimeArgsInp(diopiContextHandle_t ctx, diopiTensorHandle_t top_ks, diopiTensorHandle_t top_ps,
                                                    diopiTensorHandle_t skip_decode, int64_t batch_size, int64_t top_k, int64_t top_ks_size, float top_p,
                                                    int64_t top_ps_size);

/**
 * @brief TopKSampling. Get the new id by topk sampling.
 * 1.cal workspace_size if not.Default lmdeploy size == batch_size * vocab_size * sizeof(float) /4*4 + batch_size * max_top_k * sizeof(int) /4*4 + batch_size *
 * max_top_k * sizeof(float) /4*4
 * 2.update logit.if index >= vocab_size, then -FLT_MAX else if finished logit set to -FLT_MAX default or FLT_MAX when == end_id
 * 3.if cum_log_probs or output_log_probs is not null, if index >= vocab_size, then -FLT_MAX else if finished logit set to -FLT_MAX default or FLT_MAX when ==
 * end_id, then softmax with eps = 1e-6f
 * 4.first if skip is not null and is true, skip the batch. second if finish is not null and is true, id = endid index = index of endid
 * for the batch.Then that batch ended.
 * 5.if not in 4, find the top k logits.if cum_log_probs is null and output_log_probs is null, get temp_logits =
 * expf(logits-max_logits) as new logits for every k logits
 * 6.each batch gets rand_num = rand with their own state random * topp * sum of all k new logits.Then find the one when only one
 * left or rand_num <= 0 after updating by rand_num -= new_logit from top to smaller in k.
 * 7.ids[step*bs+bsid] = the one's index. And if cum_log_probs is not
 * null, cum_log_probs[batch_id] += logf(new_logit);if output_log_probs is not null, output_log_probs[batch_id] = logf(new_logit) - logf(sum of all k new
 * logits)
 * 8.if (sequence_length != nullptr && finished != nullptr), if not finished then sequence_length += 1.Then update finished by check the got ids == endid.
 * @param[in] ctx The diopi context.
 * @param[inout] output_ids : Output ids tensor.shape = [max_seq_len, batch_size].type = [int64, int32]
 * @param[inout] logits : Logits tensor.shape = [batch_size, vocab_size_padded].type = [float32, float16]
 * @param[inout] workspace : Workspace or buffer.shape = [workspace_size].type = [float32, float16]
 * @param[inout] workspace_size : Workspace size, if workspace_size < 0 then only cal workspace_size.type = [int64*, int32*]
 * @param[in] fusion_level : Fusion level, 0 represents no fusion, and the higher the numerical value, the higher the degree of fusion.type = [int64, int32]
 * @param[in] end_ids : End ids.shape = [batch_size].type = [int64, int32]
 * @param[inout] finished : Finished batch.shape = [batch_size].type = [bool]
 * @param[inout] sequence_lengths : Sequence lengths.shape = [batch_size].type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] vocab_size_padded : Padded vocab size.type = [int64, int32]
 * @param[in] runtime_top_k : TopKs tensor.shape = [batch_size].type = [int64, int32]
 * @param[in] runtime_top_p : TopPs tensor.shape = [batch_size].type = [float32]
 * @param[in] skip_decode : Need skip or not.shape = [batch_size].type = [bool]
 * @param[inout] cum_log_probs : Cum log probs.shape = [batch_size].type = [float32]
 * @param[inout] output_log_probs : Output log probs.shape = [batch_size].type = [float32]
 * @param[in] generators : pseudorandom number generators for sampling.shape = [batch_size]
 */
DIOPI_API diopiError_t diopiTopKSampling(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiTensorHandle_t logits, diopiTensorHandle_t workspace,
                                         int64_t* workspace_size, int64_t fusion_level, diopiConstTensorHandle_t end_ids, diopiTensorHandle_t finished,
                                         diopiTensorHandle_t sequence_lengths, int64_t step, int64_t batch_size, int64_t vocab_size_padded,
                                         diopiConstTensorHandle_t runtime_top_k, diopiConstTensorHandle_t runtime_top_p, diopiConstTensorHandle_t skip_decode,
                                         diopiTensorHandle_t cum_log_probs, diopiTensorHandle_t output_log_probs, diopiGeneratorHandle_t* generators);

/**
 * @brief SetupToppRuntimeArgs. Fix topk and topp for each batch.
 * 1.If top_ks/ps_size > 0 k/p get from top_ks/ps else top_k/p.
 * 2.If k == 0 && p == 0.0f then k = 1.
 * 3.Clip p to [0.0, 1.0] and save.
 * 4.If k > 0 means skip needed then save to skip_decode.
 * 5.Save top_ps to initial_top_p_buf.
 * 6.If top_p_decay is null or top_p_decay > 1. or top_p_decay <= 0.0f then top_p_decay_buf = 1.0f else top_p_decay_buf = top_p_decay.
 * 7.If top_p_min is null then top_p_min_buf = 1e-6f.If top_p_min > 1. or top_p_min <= 0.0f then top_p_min_buf = 0.5f.Others top_p_min_buf = top_p_min.
 * 8.If top_p_reset_ids is null then top_p_reset_ids_buf = -1 else top_p_reset_ids_buf = top_p_reset_ids.
 * @param[in] ctx The diopi context.
 * @param[inout] top_ks : TopKs tensor.shape = [batch_size].type = [int64, int32]
 * @param[inout] top_ps : TopPs tensor.shape = [batch_size].type = [float32]
 * @param[inout] skip_decode : Need skip or not.shape = [batch_size].type = [bool]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] top_k : Topk.type = [int64, int32]
 * @param[in] top_ks_size : Topks size.type = [int64, int32]
 * @param[in] top_p : Topp.type = [float32]
 * @param[in] top_ps_size : Topps size.type = [int64, int32]
 * @param[out] initial_top_p_buf : Initial_top_p_buf.shape = [batch_size].Topp.type = [float32]
 * @param[out] top_p_decay_buf : Topp decay buf.shape = [batch_size].type = [float32]
 * @param[in] top_p_decay : Topp decay.shape = [batch_size].type = [float32]
 * @param[out] top_p_min_buf : Topp min buf.shape = [batch_size].type = [float32]
 * @param[in] top_p_min : Topp min.shape = [batch_size].type = [float32]
 * @param[out] top_p_reset_ids_buf : Topp reset buf.shape = [batch_size].type = [int64, int32]
 * @param[in] top_p_reset_ids : Topp reset.shape = [batch_size].type = [int64, int32]
 */
DIOPI_API diopiError_t diopiSetupToppRuntimeArgsInp(diopiContextHandle_t ctx, diopiTensorHandle_t top_ks, diopiTensorHandle_t top_ps,
                                                    diopiTensorHandle_t skip_decode, int64_t batch_size, int64_t top_k, int64_t top_ks_size, float top_p,
                                                    int64_t top_ps_size, diopiTensorHandle_t initial_top_p_buf, diopiTensorHandle_t top_p_decay_buf,
                                                    float top_p_decay, diopiTensorHandle_t top_p_min_buf, float top_p_min,
                                                    diopiTensorHandle_t top_p_reset_ids_buf, int64_t top_p_reset_ids);

/**
 * @brief TopPSampling. Get the new id by topp sampling.
 * 1.Cal workspace_size if not.And cal persistentworkspace_size.For lmdeploy:
 * topp_id_vals_buf shape = [batch_size * vocab_size_padded].type = [int64, int32]]
 * topp_offset_buf shape = [batch_size + 1],type = [int64, int32]
 * begin_topp_offset_buf shape = [batch_size + 1],type = [int64, int32]
 * 2.Update topp_offset_buf and begin_topp_offset_buf.Set value = index * vocab_size_padded.Upate topp_id_vals_buf as flattened tensor, set value =
 * index%vocab_size_padded.
 * 3.If index >= vocab_size, then -FLT_MAX else if finished logit set to -FLT_MAX default or FLT_MAX when == end_id, then softmax with
 * eps = 1e-6f.
 * 4.get p_threshold form top_ps, if one >= p_threshold, and id is the smallest in those passed, then it is the one else find the one with
 * newp_threshold = each batch own state random * p_threshold which sum >= newp_threshold with all less then it but without it the sum is < newp_threshold.
 * 5.ids[step*bs+bsid] =
 * the one's index. And if cum_log_probs is not null, cum_log_probs[batch_id] += logf(new_logit);if output_log_probs is not null, output_log_probs[batch_id] =
 * logf(new_logit) - logf(sum of all k new logits).
 * 6.if (sequence_length != nullptr && finished != nullptr), if not finished then sequence_length += 1.Then
 * update finished by check the got ids == endid.
 * 7.if the one == top_p_reset_ids, runtime_top_p = runtime_initial_top_p else = max(runtime_top_p * top_p_decay,
 * top_p_min)
 * @param[in] ctx The diopi context.
 * @param[inout] output_ids : Output ids tensor.shape = [max_seq_len, batch_size].type = [int64, int32]
 * @param[inout] logits : Logits tensor.shape = [batch_size, vocab_size_padded].type = [float32, float16]
 * @param[inout] persistent_workspace : Persistent workspace or buffer.shape = [persistent_workspace_size].type = [int64, int32]
 * @param[inout] persistent_workspace_size : Persistent workspace size, if < 0 then only cal size.type = [int64*, int32*]
 * @param[inout] workspace : Workspace or buffer.shape = [workspace_size].type = [float32, float16]
 * @param[inout] workspace_size : Workspace size, if workspace_size < 0 then only cal workspace_size.type = [int64*, int32*]
 * @param[in] fusion_level : Fusion level, 0 represents no fusion, and the higher the numerical value, the higher the degree of fusion.type = [int64, int32]
 * @param[in] end_ids : End ids.shape = [batch_size].type = [int64, int32]
 * @param[inout] finished : Finished batch.shape = [batch_size].type = [bool]
 * @param[inout] sequence_lengths : Sequence lengths.shape = [batch_size].type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] vocab_size_padded : Padded vocab size.type = [int64, int32]
 * @param[inout] runtime_top_p : TopPs tensor.shape = [batch_size].type = [float32]
 * @param[in] skip_decode : Need skip or not.shape = [batch_size].type = [bool]
 * @param[inout] cum_log_probs : Cum log probs.shape = [batch_size].type = [float32]
 * @param[inout] output_log_probs : Output log probs.shape = [batch_size].type = [float32]
 * @param[in] generators : pseudorandom number generators for sampling.shape = [batch_size]
 */
DIOPI_API diopiError_t diopiTopPSampling(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiTensorHandle_t logits,
                                         diopiTensorHandle_t persistent_workspace, int64_t* persistent_workspace_size, diopiTensorHandle_t workspace,
                                         int64_t* workspace_size, int64_t fusion_level, diopiConstTensorHandle_t end_ids, diopiTensorHandle_t finished,
                                         diopiTensorHandle_t sequence_lengths, int64_t step, int64_t batch_size, int64_t vocab_size_padded,
                                         diopiTensorHandle_t runtime_top_p, diopiConstTensorHandle_t skip_decode, diopiTensorHandle_t cum_log_probs,
                                         diopiTensorHandle_t output_log_probs, diopiGeneratorHandle_t* generators);

/**
 * @brief GatherOutput. [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
 * src skip padding in [context_len, max_context_len) and src len <= max_gen_step
 * when src in [max_context_len, ...), dst_idx = src_idx - (max_context_len - context_len)
 * @param[in] ctx The diopi context.
 * @param[out] output_ids : Output ids.shape = [batch_size, max_output_len].type = [int64, int32]
 * @param[in] ids : Ids.shape = [session, batch_size].type = [int64, int32]
 * @param[in] context_lengths : Contextlengths.shape = [batch_size].type = [int64, int32]
 * @param[in] max_context_len : Max context len.type = [int64, int32]
 * @param[in] max_gen_step : Max gen step.type = [int64, int32]
 * @param[in] max_output_len : Max output len.type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiGatherOutput(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiConstTensorHandle_t ids,
                                         diopiConstTensorHandle_t context_length, int64_t max_context_len, int64_t max_gen_step, int64_t max_output_len,
                                         int64_t batch_size);

/**
 * @brief PlusScalar. if 0 < index < size, add val.
 * @param[in] ctx The diopi context.
 * @param[inout] inoutput : Output tensor.shape=[len].type = [int64, int32]
 * @param[in] val : Val for add.type = [int64, int32]
 * @param[in] size : Size or maxindex.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiPlusScalarInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, const int64_t val, const int64_t size);

/**
 * @brief Total_padding_count.Padding_count = maxinputlen - inputlen for each batch.
 * @param[in] ctx The diopi context.
 * @param[out] total_padding_count : Total padding_count.shape=[batch_size].type = [int64, int32]
 * @param[in] input_lengths : Input length.shape=[batch_size].type = [int64, int32]
 * @param[in] max_input_length : Max input length.type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiUpdatePaddingCount(diopiContextHandle_t ctx, diopiTensorHandle_t total_padding_count, diopiConstTensorHandle_t input_lengths,
                                               int64_t max_input_length, int64_t batch_size);

/**
 * @brief TransposeAxis01. [dim0, dim1, dim2] -> [dim1, dim0, dim2]
 * @param[in] ctx The diopi context.
 * @param[out] output : Output tensor.shape = [dim1, dim0, dim2].type = [float32, float16, int64, int32]
 * @param[in] input : Input tensor.shape = [dim0, dim1, dim2].type = [float32, float16, int64, int32]
 * @param[in] dim0 : Size of 0 dim.type = [int64, int32]
 * @param[in] dim1 : Size of 1 dim.type = [int64, int32]
 * @param[in] dim2 : Size of 2 dim.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiTransposeAxis01(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiConstTensorHandle_t input, const int64_t dim0,
                                            const int64_t dim1, const int64_t dim2);

/**
 * @brief EmbeddingLookupPosEncoding. Find id in embedding_table and get [hidden], only this step
 * @param[in] ctx The diopi context.
 * @param[out] from_tensor : Output ids.shape = [batch_size, hidden].type = [float32, float16]
 * @param[in] embedding_table : Embedding table.shape=[vocab, hidden].type = [float32, float16]
 * @param[in] all_ids : Input ids.shape=[sessionlen, batch_size].type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] hidden_units : Hidden units.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t embedding_table,
                                                       diopiConstTensorHandle_t all_ids, const int64_t batch_size, const int64_t hidden_units,
                                                       const int64_t step);

/**
 * @brief InputIdsEmbeddingLookupPosEncoding. Find id in embedding_table and get [hidden].
 * @param[in] ctx The diopi context.
 * @param[out] from_tensor : Output ids.shape = [input_lengths, hidden].type = [float32, float16]
 * @param[in] input_ids : Input ids.shape=[input_lengths].type = [int64, int32]
 * @param[in] embedding_table : Embedding table.shape=[vocab, hidden].type = [float32, float16]
 * @param[in] input_lengths : Input lengths.type = [int64, int32]
 * @param[in] hidden_units : Hidden units.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiInputIdsEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t input_ids,
                                                               diopiConstTensorHandle_t embedding_table, const int64_t input_lengths,
                                                               const int64_t hidden_units);

/**
 * @brief BanBadWords.
 * get base_bad_words and base_offsets from stop_words for each batch.
 * every item in base_offsets means item-end, and they also the item-start of the next item, for the first item item-start is 0.
 * If time-size = end - start < step+1, then check item.
 * for (int token_idx = item_size - 1; token_idx >= 0; token_idx--) {const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size +
 * id_offset + batch_idx];if (previous_token != base_stop_words[item_start + token_idx]) {should_ban = false; break;}} if this tiem should_ban, then get banid =
 * base_bad_words[item-end - 1]. if 0 < banid < vocab_size then logits ban id in this batch is set to -INFINITY
 * @param[in] ctx The diopi context.
 * @param[inout] logits : Output logits.shape = [batch_size, vocab_size].type = [float32, float16]
 * @param[in] output_ids : Output ids.shape = [batch_size, step].type = [int64, int32]
 * @param[in] bad_words : Stop words list.shape = [batch_size, 2, stop_words_len] or [2, stop_words_len] for share.type = [int64, int32]
 * @param[in] id_offset : Offset of output_ids.type = [int64, int32]
 * @param[in] bad_words_len : Stop words len.type = [int64, int32]
 * @param[in] share_words : Stop words is shared or not.type = [bool]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] vocab_size : Vocab size.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiBanBadWordsInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t output_ids,
                                           diopiConstTensorHandle_t bad_words, int64_t id_offset, int64_t bad_words_len, bool share_words, int64_t batch_size,
                                           int64_t vocab_size, int64_t step);

/**
 * @brief StopWordsCriterion. Judging the end situation based on stopword.
 * get base_stop_words and base_offsets from stop_words for each batch.
 * every item in base_offsets means item-end, and they also the item-start of the next item, for the first item item-start is 0.
 * If time-size = end - start < step+1, then check item.
 * for (int token_idx = item_size - 1; token_idx >= 0; token_idx--) {const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size +
 * id_offset + batch_idx];if (previous_token != base_stop_words[item_start + token_idx]) {should_stop = false; break;}} if one batch is should_stop, then it is
 * finished.
 * @param[in] ctx The diopi context.
 * @param[in] output_ids : Output ids.shape = [batch_size, step].type = [int64, int32]
 * @param[in] stop_words : Stop words list.shape = [batch_size, 2, stop_words_len].type = [int64, int32]
 * @param[inout] finished : Finished.shape = [batch_size].type = [bool]
 * @param[in] id_offset : Offset of output_ids.type = [int64, int32]
 * @param[in] stop_words_len : Stop words len tensor.type = [int64, int32]
 * @param[in] batch_size : batch_size.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiStopWordsCriterion(diopiContextHandle_t ctx, diopiConstTensorHandle_t output_ids, diopiConstTensorHandle_t stop_words,
                                               diopiTensorHandle_t finished, int64_t id_offset, int64_t stop_words_len, int64_t batch_size, int64_t step);

/**
 * @brief LengthCriterion. Judging and counting the end situation based on length.If all fin then should_stop.
 * @param[in] ctx The diopi context.
 * @param[inout] finished : Finished.shape = [batch_size].type = [bool]
 * @param[out] should_stop : If all fin then should_stop.shape = [1].type = [bool]
 * @param[out] finished_sum : Total finished.shape = [1].type = [int64, int32]
 * @param[in] sequence_limit_length : Sequence limit length tensor.shape = [batch_size].type = [int64, int32]
 * @param[in] batch_size : Input tensor.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiLengthCriterion(diopiContextHandle_t ctx, diopiTensorHandle_t finished, diopiTensorHandle_t should_stop,
                                            diopiTensorHandle_t finished_sum, diopiConstTensorHandle_t sequence_limit_length, int64_t batch_size, int64_t step);

/**
 * @brief Apply repetitionPenalty.If 0 <= index < step with (if index >= input_length && index < max_input_length then continue), save every index and fixed
 * logit in penalty_logits and penalty_indices with shape [step].Then update logits with penalty_logits and penalty_indices.
 * @param[in] ctx The diopi context.
 * @param[inout] logits : nput tensor logits.shape = [batch_size, vocab_size].type = [float32, float16]
 * @param[in] penalties : Penalties tensor.shape = [batch_size].type = [float32]
 * @param[in] output_ids : The bias tensor.shape = [max_seq_len or step, batch_size].type = [int64, int32].
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] vocab_size : Vocab size.shape = [batch_size].type = [int64, int32]
 * @param[in] input_lengths : Input lengths tensor.shape = [batch_size].type = [int64, int32]
 * @param[in] max_input_length : Max input length.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 * @param[in] penalty_type : Penalty type.0 == None;1 == Additive means logit - penalty;2 == Multiplicative means logit < 0.0f ? logit * penalty : logit /
 * penalty.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiBatchApplyRepetitionPenaltyInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t penalties,
                                                           diopiConstTensorHandle_t output_ids, const int64_t batch_size, const int64_t vocab_size,
                                                           diopiConstTensorHandle_t input_lengths, const int64_t max_input_length, const int64_t step,
                                                           const int64_t penalty_type);

/**
 * @brief Apply temperaturePenalty with bias. bias_val = bias[index in vocab_size_padd] and if index in vocab_size then logits = (logits +
 * bias_val)*(1.0f/(temperature + 1e-6f)) else logits = -FLT_MAX
 * @param[in] ctx The diopi context.
 * @param[inout] logits : Output tensor logits.shape = [batch_size, vocab_size_padded].type = [float32, float16]
 * @param[in] bias : Input tensor bias.shape = [vocab_size_padded].type = [float32, float16]
 * @param[in] temperatures : Temperatures.shape = [batch_size].type = [float32].
 * @param[in] batch_size : Batch size.type = [int64, int32].
 * @param[in] vocab_size : Vocab size.type = [int64, int32].
 * @param[in] vocab_size_padd : Vocab padd size.type = [int64, int32].
 */
DIOPI_API diopiError_t diopiBatchApplyTemperaturePenaltyInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t bias,
                                                            diopiConstTensorHandle_t temperatures, const int64_t batch_size, const int64_t vocab_size,
                                                            const int64_t vocab_size_padd);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LMDEPLOY_H_