/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <ATen/core/ATen_fwd.h>
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
    TORCH_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4 dimensional, but got ", qAt.dim(), "-dimensional");
    TORCH_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4 dimensional, but got ", kAt.dim(), "-dimensional");
    TORCH_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4 dimensional, but got ", vAt.dim(), "-dimensional");
    at::Tensor realShiftOptional;
    at::Tensor dropMaskOptional;
    at::Tensor paddingMaskOptional;
    at::Tensor attentionMaskOptional = attention_maskAt;
    auto prefixOptional = nullptr;
    double scaleValueOptional = softmax_scale;
    double keepProbOptional = 1 - p_dropout;
    int64_t nextTockensOptional = 0;
    const int64_t innerPreciseOptional = 0;
    int64_t sparseModeOptional = 0;
    const char* inputLayout = "BSND";
    const int64_t B = qAt.size(0);
    const int64_t Sq = qAt.size(1);
    const int64_t Sk = kAt.size(1);
    const int64_t headNum = qAt.size(2);
    const int64_t N = headNum;
    const int64_t preTockensOptional = Sq + Sk;

    const auto qShape = qAt.sizes();
    std::vector<int64_t> softmaxMaxShape{B, N, Sq, 8};  // [B, N, Sq, 8]
    at::Tensor softmaxMaxOut = at_npu::native::empty_npu(softmaxMaxShape, attention_outAt.options().dtype(at::kFloat));
    at::Tensor softmaxSumOut = at_npu::native::empty_npu(softmaxMaxShape, attention_outAt.options().dtype(at::kFloat));
    at::Tensor softmaxOutOut = at_npu::native::empty_npu({0}, attention_outAt.options().dtype(at::kFloat));
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

diopiError_t diopiAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                    diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                    diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t* saved_for_backward, int64_t saved_tensor_num,
                                    double pDropout, diopiGeneratorHandle_t gen_dropout, double softmaxScale, const char* attention_type) {
    BEGIN_CALL_ACL_OP(q, k, v, attentionOut, gradQ, gradK, gradV, gradOut);

    TORCH_CHECK(saved_tensor_num >= 5, "backward need 5 tensors saved in forward")
    const at::Tensor softmaxMaxAt = impl::aten::buildATen(saved_for_backward[0]);
    const at::Tensor softmaxSumAt = impl::aten::buildATen(saved_for_backward[1]);
    const at::Tensor softmaxOutAt = impl::aten::buildATen(saved_for_backward[2]);
    const at::Tensor attentionMaskAt = impl::aten::buildATen(saved_for_backward[3]);
    const at::Tensor dropoutMaskAt = impl::aten::buildATen(saved_for_backward[4]);

    DIOPI_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4-dimensional");
    DIOPI_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4-dimensional");
    DIOPI_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    const char* inputLayout = "BSND";
    int64_t headNum = qAt.size(2);
    int64_t Sk = kAt.size(1);
    int64_t Sq = kAt.size(1);
    double keepProb = 1 - pDropout;

    at::Tensor pseAt;
    at::Tensor gradPseAt = at_npu::native::OpPreparation::apply_tensor_without_format({0}, qAt.options());
    at::IntArrayRef prefixN;

    at::Tensor paddingMaskAt;

    int64_t preTokens = Sq + Sk;
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScoreGrad,
                                 qAt,
                                 kAt,
                                 vAt,
                                 gradOutAt,
                                 pseAt,
                                 dropoutMaskAt,
                                 paddingMaskAt,
                                 attentionMaskAt,
                                 softmaxMaxAt,
                                 softmaxSumAt,
                                 softmaxOutAt,
                                 attentionOutAt,
                                 prefixN,
                                 softmaxScale,
                                 keepProb,
                                 preTokens,
                                 nextTokens,
                                 headNum,
                                 inputLayout,
                                 innerPrecise,
                                 sparseMode,
                                 gradQAt,
                                 gradKAt,
                                 gradVAt,
                                 gradPseAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
