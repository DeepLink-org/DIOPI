/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */
#include <ATen/core/ATen_fwd.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstdint>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* saveForBackward, int64_t* saveTensorNum,
                                  diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t cuSeqlensQ,
                                  diopiConstTensorHandle_t cuSeqlensKv, int64_t maxSeqlenQ, int64_t maxKvLenKv, diopiConstTensorHandle_t attentionMask,
                                  diopiConstTensorHandle_t attentionBias, double pDropout, diopiGeneratorHandle_t genDropout, double softmaxScale,
                                  bool isCausal, const char* attentionType) {
    BEGIN_CALL_ACL_OP(attentionOut, q, k, v, cuSeqlensQ, cuSeqlensKv, attentionMask, attentionBias, genDropout);
    TORCH_CHECK(qAt.dim() == 3, "The shapes of the input query should be 3 dimensional, but got ", qAt.dim(), "-dimensional");
    TORCH_CHECK(kAt.dim() == 3, "The shapes of the input key should be 3 dimensional, but got ", kAt.dim(), "-dimensional");
    TORCH_CHECK(vAt.dim() == 3, "The shapes of the input value should be 3 dimensional, but got ", vAt.dim(), "-dimensional");
    TORCH_CHECK(cuSeqlensKvAt.numel() == cuSeqlensQAt.numel(), "q k shoule have same batchsize");
    TORCH_CHECK(cuSeqlensKvAt.scalar_type() == at::kLong);
    TORCH_CHECK(cuSeqlensQAt.scalar_type() == at::kLong);
    const int64_t totalSeqQ = qAt.size(0);
    const int64_t totalSeqK = kAt.size(0);
    const int64_t headNum = qAt.size(1);
    const int64_t headDim = qAt.size(2);
    const int64_t batch_size = cuSeqlensQAt.sizes().size() - 1;
    at::Tensor pse;
    at::Tensor dropMaskOptional;
    at::Tensor paddingMaskOptional;
    at::Tensor attentionMaskOptional = attentionMaskAt;
    auto prefixOptional = nullptr;
    double scaleValueOptional = softmaxScale;
    double keepProbOptional = 1 - pDropout;
    const int64_t preTockensOptional = totalSeqQ;
    int64_t nextTockensOptional = 0;
    const int64_t innerPreciseOptional = 0;
    int64_t sparseModeOptional = 0;
    const char* inputLayout = "TND";

    const auto qShape = qAt.sizes();
    std::vector<int64_t> softmaxMaxShape{totalSeqQ, headNum, 8};  // [T, N, 8]
    at::Tensor softmaxMaxOut = at_npu::native::empty_npu(softmaxMaxShape, attentionOutAt.options().dtype(at::kFloat));
    at::Tensor softmaxSumOut = at_npu::native::empty_npu(softmaxMaxShape, attentionOutAt.options().dtype(at::kFloat));
    at::Tensor softmaxOutOut = at_npu::native::empty_npu({0}, attentionOutAt.options().dtype(at::kFloat));
    if (attentionBiasAt.defined()) {
        pse = attentionBiasAt;
    }
    if (isCausal) {
        attentionMaskOptional = at_npu::native::empty_npu({maxSeqlenQ, maxKvLenKv}, qAt.options().dtype(at::kBool));  // [maxSq, maxSk]
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
        int64_t numels = totalSeqK * totalSeqQ * headNum;
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
    at::Tensor cuSeqlensKvAtHost = cuSeqlensKvAt.cpu();
    at::Tensor cuSeqlensQAtHost = cuSeqlensQAt.cpu();
    at::IntArrayRef cuSeqlensKvAtArray(cuSeqlensKvAtHost.data_ptr<int64_t>(), cuSeqlensKvAtHost.numel());
    at::IntArrayRef cuSeqlensQAtArray(cuSeqlensQAtHost.data_ptr<int64_t>(), cuSeqlensQAtHost.numel());
    EXEC_NPU_CMD(aclnnFlashAttentionVarLenScore,
                 qAt,
                 kAt,
                 vAt,
                 pse,
                 dropMaskOptional,
                 paddingMaskOptional,
                 attentionMaskOptional,
                 prefixOptional,
                 cuSeqlensQAtArray,
                 cuSeqlensKvAtArray,
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

}  // namespace OP_IMPL_NS
