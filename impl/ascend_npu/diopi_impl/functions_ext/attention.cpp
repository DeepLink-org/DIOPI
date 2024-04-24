/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <csrc/framework/DIOPIAdapter.h>

#include <cstdint>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t* save_for_backward, int64_t* save_tensor_num,
                            diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t attention_mask,
                            double p_dropout, diopiGeneratorHandle_t gen_dropout, double softmax_scale, bool is_causal, const char* attention_type) {
    BEGIN_CALL_ACL_OP(attention_out, q, k, v, attention_mask, gen_dropout);
    TORCH_CHECK(qAt.dim() == 4 || qAt.dim() == 3, "The shapes of the input query should be 4 or 3 dimensional, but got ", qAt.dim(), "-dimensional");
    TORCH_CHECK(kAt.dim() == 4 || kAt.dim() == 3, "The shapes of the input key should be 4 or 3 dimensional, but got ", kAt.dim(), "-dimensional");
    TORCH_CHECK(vAt.dim() == 4 || vAt.dim() == 3, "The shapes of the input value should be 4 or 3 dimensional, but got ", vAt.dim(), "-dimensional");
    at::Tensor realShiftOptional;
    at::Tensor dropMaskOptional;
    at::Tensor paddingMaskOptional;
    at::Tensor attentionMaskOptional = attention_maskAt;
    auto prefixOptional = nullptr;
    double scaleValueOptional = softmax_scale;
    double keepProbOptional = 1 - p_dropout;
    int64_t nextTockensOptional = 0;
    const int64_t innerPreciseOptional = 0;
    const int64_t sparseModeOptional = 0;
    const bool is_4d_input = qAt.sizes().size() == 4;
    const char* inputLayout = is_4d_input ? "BNSD" : "BSH";
    const int64_t headNum = is_4d_input ? qAt.size(1) : 1;
    const int64_t B = qAt.size(0);
    const int64_t Sq = is_4d_input ? qAt.size(2) : qAt.size(1);
    const int64_t Sk = is_4d_input ? kAt.size(2) : kAt.size(1);
    const int64_t N = headNum;
    const int64_t preTockensOptional = Sk;

    const auto qShape = qAt.sizes();
    std::vector<int64_t> softmaxMaxShape{B, N, Sq, 8};  // [B, N, Sq, 8]

    at::Tensor softmaxMaxOut = at_npu::native::empty_npu(softmaxMaxShape, attention_outAt.options().dtype(at::kFloat));
    at::Tensor softmaxSumOut = at_npu::native::empty_npu(softmaxMaxShape, attention_outAt.options().dtype(at::kFloat));
    at::Tensor softmaxOutOut = at_npu::native::empty_npu({1}, attention_outAt.options().dtype(at::kFloat));
    if (is_causal) {
        attentionMaskOptional = at_npu::native::empty_npu({Sq, Sk}, qAt.options().dtype(at::kBool));  // [Sq, Sk]
        EXEC_NPU_CMD(aclnnInplaceOne, attentionMaskOptional);
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskOptional, diagonal);
    }

    TORCH_CHECK(keepProbOptional >= 0 && keepProbOptional <= 1, "The keep_prob value must be in range of [0, 1], but got ", keepProbOptional);
    TORCH_CHECK(sparseModeOptional >= 0 && sparseModeOptional <= 5, "The sparse_mode value must be in range of [0~5], but got ", sparseModeOptional);

    if (p_dropout > 0 && p_dropout <= 1) {
        int64_t numels = B * N * Sq * Sk;  // [B,N,S,S]
        constexpr int64_t bitNumber = 128;
        constexpr int64_t uInt8BitNumber = 8;
        int64_t length = (numels + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        dropMaskOptional = at_npu::native::empty_npu({length}, qAt.options().dtype(at::kByte));
        if (p_dropout == 1) {
            op_api::zero_(dropMaskOptional);
        } else {
            std::vector<int64_t> shapeVector{numels};
            at::IntArrayRef shapeArray(shapeVector);
            auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen_dropoutAt)->philox_engine_inputs(10);
            const uint64_t seed = pair.first;
            const uint64_t offset = pair.second;
            EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, p_dropout, seed, offset, dropMaskOptional);
        }
    }

    EXEC_NPU_CMD(aclnnFlashAttentionScore,
                 qAt,
                 kAt,
                 vAt,
                 realShiftOptional,
                 dropMaskOptional,
                 paddingMaskOptional,
                 attentionMaskOptional,
                 prefixOptional,
                 scaleValueOptional,
                 keepProbOptional,
                 preTockensOptional,
                 nextTockensOptional,
                 headNum,
                 inputLayout,
                 innerPreciseOptional,
                 sparseModeOptional,
                 softmaxMaxOut,
                 softmaxSumOut,
                 softmaxOutOut,
                 attention_outAt);
    save_for_backward[0] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxMaxOut)->npu_desc_.diopi_tensor_;
    save_for_backward[1] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxSumOut)->npu_desc_.diopi_tensor_;
    save_for_backward[2] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxOutOut)->npu_desc_.diopi_tensor_;
    save_for_backward[3] = attentionMaskOptional.defined() ? torch_npu::NPUBridge::GetNpuStorageImpl(attentionMaskOptional)->npu_desc_.diopi_tensor_ : nullptr;
    save_for_backward[4] = dropMaskOptional.defined() ? torch_npu::NPUBridge::GetNpuStorageImpl(dropMaskOptional)->npu_desc_.diopi_tensor_ : nullptr;
    DEBUG_ARGS(dropMaskOptional);
    *save_tensor_num = 5;
    END_CALL_ACL_OP();
}

diopiError_t diopiAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v,
                                    diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                    diopiConstTensorHandle_t attention_out, diopiConstTensorHandle_t* saved_for_backward, int64_t saved_tensor_num,
                                    double p_dropout, diopiGeneratorHandle_t gen_dropout, double softmax_scale, bool is_causal, const char* attention_type) {
    BEGIN_CALL_ACL_OP(attention_out, q, k, v, grad_q, grad_k, grad_v, grad_out, gen_dropout);
    TORCH_CHECK(saved_tensor_num >= 4, "backward need 4 tensors saved in forward, bug got ", saved_tensor_num)
    const at::Tensor softmaxMaxOptional = impl::aten::buildATen(saved_for_backward[0]);
    const at::Tensor softmaxSumOptional = impl::aten::buildATen(saved_for_backward[1]);
    const at::Tensor softmaxInOptional = impl::aten::buildATen(saved_for_backward[2]);
    const at::Tensor attentionMaskOptional = impl::aten::buildATen(saved_for_backward[3]);
    const at::Tensor dropMaskOptional = impl::aten::buildATen(saved_for_backward[4]);

    DEBUG_ARGS(softmaxMaxOptional);
    DEBUG_ARGS(softmaxSumOptional);
    DEBUG_ARGS(softmaxInOptional);
    DEBUG_ARGS(attentionMaskOptional);
    DEBUG_ARGS(dropMaskOptional);

    at::Tensor prefixOptional;
    double scaleValueOptional = softmax_scale;
    double keepProbOptional = 1 - p_dropout;
    int64_t preTockensOptional = kAt.size(1);
    int64_t nextTockensOptional = 0;
    const bool is_4d_input = qAt.sizes().size() == 4;
    const char* inputLayout = is_4d_input ? "BNSD" : "BSH";
    const int64_t headNum = is_4d_input ? qAt.size(1) : 1;
    int64_t innerPreciseOptional = 0;
    int64_t sparseModeOptional = 0;
    at::Tensor dpseOut = at_npu::native::empty_npu(vAt.sizes(), vAt.options());
    at::Tensor gradPse = at_npu::native::empty_npu({1}, vAt.options());
    at::Tensor pseShiftOptional;
    at::Tensor paddingMaskOptional;
    EXEC_NPU_CMD(aclnnFlashAttentionScoreGrad,
                 qAt,
                 kAt,
                 vAt,
                 grad_outAt,
                 pseShiftOptional,
                 dropMaskOptional,
                 paddingMaskOptional,
                 attentionMaskOptional,
                 attention_outAt,
                 prefixOptional,
                 scaleValueOptional,
                 keepProbOptional,
                 preTockensOptional,
                 nextTockensOptional,
                 headNum,
                 inputLayout,
                 innerPreciseOptional,
                 sparseModeOptional,
                 grad_qAt,
                 grad_kAt,
                 grad_vAt,
                 gradPse);
    END_CALL_ACL_OP();
}

diopiError_t diopiAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attention_out, diopiTensorHandle_t* save_for_backward, int64_t* save_tensor_num,
                                  diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t attention_mask,
                                  double p_dropout, diopiGeneratorHandle_t gen_dropout, diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k,
                                  int64_t max_seq_length, double softmax_scale, bool is_causal, const char* attention_type) {
    BEGIN_CALL_ACL_OP(attention_out, q, k, v, attention_mask, gen_dropout);
    at::Tensor realShiftOptional;
    at::Tensor dropMaskOptional;
    at::Tensor paddingMaskOptional;
    at::Tensor attentionMaskOptional = attention_maskAt;
    auto prefixOptional = nullptr;
    double scaleValueOptional = softmax_scale;
    double keepProbOptional = 1 - p_dropout;
    int64_t preTockensOptional = 0;
    int64_t nextTockensOptional = 0;
    int64_t headNum = qAt.size(1);
    const char* inputLayout = qAt.sizes().size() == 4 ? "BNSD" : "BSH";
    int64_t innerPreciseOptional = 0;
    int64_t sparseModeOptional = 0;
    at::Tensor softmaxMaxOut = at_npu::native::empty_npu({1}, attention_outAt.options().dtype(at::kFloat));
    at::Tensor softmaxSumOut = at_npu::native::empty_npu({1}, attention_outAt.options().dtype(at::kFloat));
    at::Tensor softmaxOutOut = at_npu::native::empty_npu({1}, attention_outAt.options().dtype(at::kFloat));
    if (is_causal) {
        int64_t s0 = qAt.size(-2);                                                                    // S for query
        int64_t s1 = kAt.size(-2);                                                                    // S for key & value
        attentionMaskOptional = at_npu::native::empty_npu({s0, s1}, qAt.options().dtype(at::kBool));  // [S0, S1]
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskOptional, diagonal);
    }
    TORCH_CHECK(qAt.dim() == 3 || qAt.dim() == 4, "The shapes of the input query should be 3 or 4 dimensional, but got ", qAt.dim(), "-dimensional");
    TORCH_CHECK(kAt.dim() == 3 || kAt.dim() == 4, "The shapes of the input key should be 3 or 4 dimensional, but got ", kAt.dim(), "-dimensional");
    TORCH_CHECK(vAt.dim() == 3 || vAt.dim() == 4, "The shapes of the input value should be 3 or 4 dimensional, but got ", vAt.dim(), "-dimensional");
    TORCH_CHECK(keepProbOptional >= 0 && keepProbOptional <= 1, "The keep_prob value must be in range of [0, 1], but got ", keepProbOptional);
    TORCH_CHECK(sparseModeOptional >= 0 && sparseModeOptional <= 5, "The sparse_mode value must be in range of [0~5], but got ", sparseModeOptional);
    EXEC_NPU_CMD(aclnnFlashAttentionScore,
                 qAt,
                 kAt,
                 vAt,
                 realShiftOptional,
                 dropMaskOptional,
                 paddingMaskOptional,
                 attentionMaskOptional,
                 prefixOptional,
                 scaleValueOptional,
                 keepProbOptional,
                 prefixOptional,
                 nextTockensOptional,
                 headNum,
                 inputLayout,
                 innerPreciseOptional,
                 sparseModeOptional,
                 softmaxMaxOut,
                 softmaxSumOut,
                 softmaxOutOut,
                 attention_outAt);
    save_for_backward[0] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxMaxOut)->npu_desc_.diopi_tensor_;
    save_for_backward[1] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxSumOut)->npu_desc_.diopi_tensor_;
    save_for_backward[2] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxOutOut)->npu_desc_.diopi_tensor_;
    *save_tensor_num = 3;
    END_CALL_ACL_OP();
}

diopiError_t diopiAttentionVarLenBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_q, diopiTensorHandle_t grad_k, diopiTensorHandle_t grad_v,
                                          diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                          diopiConstTensorHandle_t cum_seq_q, diopiConstTensorHandle_t cum_seq_k, diopiConstTensorHandle_t attention_out,
                                          diopiConstTensorHandle_t attention_mask, diopiConstTensorHandle_t* saved_for_backward, int64_t saved_tensor_num,
                                          int64_t max_seq_length, double p_dropout, diopiGeneratorHandle_t gen_dropout, double softmax_scale, bool is_causal,
                                          const char* attention_type) {
    BEGIN_CALL_ACL_OP(attention_out, q, k, v, grad_q, grad_k, grad_v, grad_out, attention_mask, gen_dropout);
    at::Tensor pseShiftOptional;
    at::Tensor dropMaskOptional;
    at::Tensor paddingMaskOptional;
    TORCH_CHECK(saved_tensor_num >= 3, "backward need 3 tensors")
    at::Tensor softmaxMaxOptional = impl::aten::buildATen(saved_for_backward[0]);
    at::Tensor softmaxSumOptional = impl::aten::buildATen(saved_for_backward[1]);
    at::Tensor softmaxInOptional = impl::aten::buildATen(saved_for_backward[2]);
    at::Tensor prefixOptional;
    double scaleValueOptional = softmax_scale;
    double keepProbOptional = 1 - p_dropout;
    int64_t preTockensOptional = 0;
    int64_t nextTockensOptional = 0;
    int64_t headNum = qAt.size(1);
    const char* inputLayout = qAt.sizes().size() == 4 ? "BNSD" : "BSH";
    int64_t innerPreciseOptional = 0;
    int64_t sparseModeOptional = 0;
    at::Tensor dpseOut = at_npu::native::empty_npu(vAt.sizes(), vAt.options());
    EXEC_NPU_CMD(aclnnFlashAttentionScoreGrad,
                 qAt,
                 kAt,
                 vAt,
                 grad_outAt,
                 pseShiftOptional,
                 dropMaskOptional,
                 paddingMaskOptional,
                 attention_maskAt,
                 attention_outAt,
                 prefixOptional,
                 scaleValueOptional,
                 keepProbOptional,
                 preTockensOptional,
                 nextTockensOptional,
                 headNum,
                 inputLayout,
                 innerPreciseOptional,
                 sparseModeOptional,
                 grad_qAt,
                 grad_kAt,
                 grad_vAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
