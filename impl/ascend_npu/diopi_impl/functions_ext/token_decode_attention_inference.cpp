/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiTokenDecodeAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc, diopiConstTensorHandle_t b_seq_len,
                                                int max_input_len, int other_kv_index) {
    BEGIN_CALL_ACL_OP(out, q, k, v, b_loc, b_start_loc, b_seq_len);
    int batch = b_locAt.size(0);
    int head_num_q = qAt.size(1);
    int dim = qAt.size(2);
    int hidden_size_q = head_num_q * dim;
    int head_num_kv = kAt.size(1);
    int hidden_size_kv = head_num_kv * dim;
    double scaleValue = 1. / std::sqrt(dim);
    qAt = qAt.reshape({batch, 1, hidden_size_q});
    c10::ScalarType dtype = qAt.scalar_type();
    c10::Device device = qAt.device();
    c10::Layout layout = qAt.layout();
    at::Tensor bSeqLenCpu = b_seq_lenAt.cpu();
    at::Tensor bStartLocCpu = b_start_locAt.cpu();
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = bSeqLenCpu[i].item<int>();
        int curSeqStartLoc = bStartLocCpu[i].item<int>();
        at::Tensor kvLoc = at::index_select(b_locAt[i], 0, acl_op::arange(max_input_len - curSeqLen, max_input_len, at::kInt, layout, device, false));
        at::Tensor key = at::index(kAt, {kvLoc}).view({1, curSeqLen, hidden_size_kv});
        at::Tensor value = at::index(vAt, {kvLoc}).view({1, curSeqLen, hidden_size_kv});
        at::Tensor query = at::index(qAt, {torch::scalar_to_tensor(i)});
        auto attentionOut = op_api::npu_incre_flash_attention(query, key, value, c10::nullopt, c10::nullopt, c10::nullopt, head_num_q, scaleValue, "BSH", head_num_kv);
        at::index_put_(outAt, {torch::scalar_to_tensor(i)}, attentionOut.reshape({head_num_q, dim}));
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiTokenDecodeAttentionInferenceBatchOne(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                                        diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                                        diopiConstTensorHandle_t b_loc, diopiConstTensorHandle_t b_start_loc, diopiConstTensorHandle_t b_seq_len,
                                                        int max_input_len, int other_kv_index) {
    BEGIN_CALL_ACL_OP(out, q, k, v, b_loc, b_start_loc, b_seq_len);
    int head_num_q = qAt.size(1);
    int dim = qAt.size(2);
    int hidden_size_q = head_num_q * dim;
    int head_num_kv = kAt.size(1);
    int hidden_size_kv = head_num_kv * dim;
    double scaleValue = 1. / std::sqrt(dim);
    qAt = qAt.reshape({1, 1, hidden_size_q});
    at::Tensor bSeqLenCpu = b_seq_lenAt.cpu();
    int curSeqLen = bSeqLenCpu[0].item<int>();
    at::Tensor key = at::slice(kAt, 0, 0, curSeqLen, 1).view({1, curSeqLen, hidden_size_kv});
    at::Tensor value = at::slice(vAt, 0, 0, curSeqLen, 1).view({1, curSeqLen, hidden_size_kv});
    auto attentionOut = op_api::npu_incre_flash_attention(qAt, key, value, c10::nullopt, c10::nullopt, c10::nullopt, head_num_q, scaleValue, "BSH", head_num_kv);
    outAt.copy_(attentionOut.reshape({1, head_num_q, dim}));
    END_CALL_ACL_OP();
}

diopiError_t diopiIncreFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q,
                                      diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t paddingMask, diopiConstTensorHandle_t attenMask,
                                      diopiSize_t actualSeqLengths, int64_t numHeads, double scaleValue, const char* inputLayout, int64_t numKeyValueHeads) {
    BEGIN_CALL_ACL_OP(out, q, k, v, paddingMask, attenMask, actualSeqLengths);
    at::Tensor result = op_api::npu_incre_flash_attention(qAt, kAt, vAt, paddingMaskAt, attenMaskAt, actualSeqLengthsAt, numHeads, scaleValue, inputLayout, numKeyValueHeads);
    outAt.copy_(result);
    END_CALL_ACL_OP()
}

}  // namespace OP_IMPL_NS
