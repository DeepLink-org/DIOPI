/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */
#include <c10/core/ScalarType.h>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* saveForBackward, int64_t* saveTensorNum,
                            diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t attentionMask,
                            diopiConstTensorHandle_t attentionBias, double pDropout, diopiGeneratorHandle_t genDropout, double softmaxScale, bool isCausal) {
    BEGIN_CALL_ACL_OP(attentionOut, q, k, v, attentionMask, attentionBias, genDropout);
    TORCH_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4 dimensional, but got ", qAt.dim(), "-dimensional");
    TORCH_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4 dimensional, but got ", kAt.dim(), "-dimensional");
    TORCH_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4 dimensional, but got ", vAt.dim(), "-dimensional");
    at::Tensor pse;
    at::Tensor dropMaskOptional;
    at::Tensor paddingMaskOptional;
    at::Tensor attentionMaskOptional = attentionMaskAt;
    auto prefixOptional = nullptr;
    double scaleValueOptional = softmaxScale;
    double keepProbOptional = 1 - pDropout;
    int64_t nextTockensOptional = 0;
    const int64_t innerPreciseOptional = 0;
    int64_t sparseModeOptional = 0;
    const char* inputLayout = "BSND";
    const int64_t batch = qAt.size(0);
    const int64_t sq = qAt.size(1);
    const int64_t sk = kAt.size(1);
    const int64_t headNum = qAt.size(2);
    const int64_t preTockensOptional = sq + sk;

    const auto qShape = qAt.sizes();
    std::vector<int64_t> softmaxMaxShape{batch, headNum, sq, 8};  // [B, N, sq, 8]
    at::Tensor softmaxMaxOut = at_npu::native::empty_npu(softmaxMaxShape, attentionOutAt.options().dtype(at::kFloat));
    at::Tensor softmaxSumOut = at_npu::native::empty_npu(softmaxMaxShape, attentionOutAt.options().dtype(at::kFloat));
    at::Tensor softmaxOutOut = at_npu::native::empty_npu({0}, attentionOutAt.options().dtype(at::kFloat));
    if (isCausal) {
        attentionMaskOptional = at_npu::native::empty_npu({sq, sk}, qAt.options().dtype(at::kBool));  // [sq, sk]
        EXEC_NPU_CMD(aclnnInplaceOne, attentionMaskOptional);
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskOptional, diagonal);
    }
    if (attentionBiasAt.defined()) {
        pse = attentionBiasAt;
    }

    TORCH_CHECK(keepProbOptional >= 0 && keepProbOptional <= 1, "The keep_prob value must be in range of [0, 1], but got ", keepProbOptional);
    TORCH_CHECK(sparseModeOptional >= 0 && sparseModeOptional <= 5, "The sparse_mode value must be in range of [0~5], but got ", sparseModeOptional);

    if (pDropout > 0 && pDropout <= 1) {
        int64_t numels = batch * headNum * sq * sk;  // [B,N,S,S]
        constexpr int64_t bitNumber = 128;
        constexpr int64_t uInt8BitNumber = 8;
        int64_t length = (numels + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        dropMaskOptional = at_npu::native::empty_npu({length}, qAt.options().dtype(at::kByte));
        if (pDropout == 1) {
            op_api::zero_(dropMaskOptional);
        } else {
            std::vector<int64_t> shapeVector{numels};
            at::IntArrayRef shapeArray(shapeVector);
            auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(genDropoutAt)->philox_engine_inputs(10);
            const uint64_t seed = pair.first;
            const uint64_t offset = pair.second;
            EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, pDropout, seed, offset, dropMaskOptional);
        }
    }

    EXEC_NPU_CMD(aclnnFlashAttentionScore,
                 qAt,
                 kAt,
                 vAt,
                 pse,
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
                 attentionOutAt);
    saveForBackward[0] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxMaxOut)->npu_desc_.diopi_tensor_;
    saveForBackward[1] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxSumOut)->npu_desc_.diopi_tensor_;
    saveForBackward[2] = torch_npu::NPUBridge::GetNpuStorageImpl(softmaxOutOut)->npu_desc_.diopi_tensor_;
    saveForBackward[3] = attentionMaskOptional.defined() ? torch_npu::NPUBridge::GetNpuStorageImpl(attentionMaskOptional)->npu_desc_.diopi_tensor_ : nullptr;
    saveForBackward[4] = dropMaskOptional.defined() ? torch_npu::NPUBridge::GetNpuStorageImpl(dropMaskOptional)->npu_desc_.diopi_tensor_ : nullptr;
    saveForBackward[5] = attentionBiasAt.defined() ? torch_npu::NPUBridge::GetNpuStorageImpl(attentionBiasAt)->npu_desc_.diopi_tensor_ : nullptr;
    DEBUG_ARGS(dropMaskOptional);
    *saveTensorNum = 6;
    END_CALL_ACL_OP();
}

diopiError_t diopiAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                    diopiTensorHandle_t gradAttnBias, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                    diopiConstTensorHandle_t v, diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t* savedForBackward,
                                    int64_t savedTensorNum, double pDropout, diopiGeneratorHandle_t genDropout, double softmaxScale) {
    BEGIN_CALL_ACL_OP(q, k, v, attentionOut, gradQ, gradK, gradV, gradAttnBias, gradOut);

    TORCH_CHECK(savedTensorNum >= 6, "backward need 6 tensors saved in forward");
    const at::Tensor softmaxMaxAt = impl::aten::buildATen(savedForBackward[0]);
    const at::Tensor softmaxSumAt = impl::aten::buildATen(savedForBackward[1]);
    const at::Tensor softmaxOutAt = impl::aten::buildATen(savedForBackward[2]);
    const at::Tensor attentionMaskAt = impl::aten::buildATen(savedForBackward[3]);
    const at::Tensor dropoutMaskAt = impl::aten::buildATen(savedForBackward[4]);
    const at::Tensor pseAt = impl::aten::buildATen(savedForBackward[5]);

    DIOPI_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4-dimensional");
    DIOPI_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4-dimensional");
    DIOPI_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The pDropout value must be in range of [0, 1]");

    const char* inputLayout = "BSND";
    int64_t headNum = qAt.size(2);
    int64_t sk = kAt.size(1);
    int64_t sq = qAt.size(1);
    double keepProb = 1 - pDropout;

    at::Tensor gradPseAt;
    if (gradAttnBiasAt.defined()) {
        // todo: support shape broadcast
        gradPseAt = gradAttnBiasAt;
    } else {
        gradPseAt = at_npu::native::OpPreparation::apply_tensor_without_format({0}, qAt.options());
    }
    at::IntArrayRef prefixN;

    at::Tensor paddingMaskAt;

    int64_t preTokens = sq + sk;
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
