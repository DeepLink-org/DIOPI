/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

namespace {

using npu_preparation = at_npu::native::OpPreparation;

const int64_t bitNumber = 128;
const int64_t uInt8BitNumber = 8;

diopiError_t dropoutGenMask(at::Tensor& mask, const at::Tensor& input, double p, at::Generator gen) {
    at::IntArrayRef shapeArray(input.sizes());
    auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
    const uint64_t seed = pair.first;
    const uint64_t offset = pair.second;
    EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, p, seed, offset, mask);
    return diopiSuccess;
}

}  // namespace

diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                                 diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal) {
    BEGIN_CALL_ACL_OP(q, k, v, gen, attentionOut);

    DIOPI_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4-dimensional");
    DIOPI_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4-dimensional");
    DIOPI_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    std::string inputLayout = "BSND";
    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());

    int64_t b = qAt.size(0);
    int64_t s0 = qAt.size(1);  // S for query
    int64_t s1 = kAt.size(1);  // S for key & value
    int64_t n = qAt.size(2);
    int64_t d = qAt.size(3);

    double keepProb = 1 - pDropout;

    at::Tensor pseAt = at::Tensor();
    at::IntArrayRef prefixN = at::IntArrayRef{};

    at::Tensor paddingMaskAt = at::Tensor();

    at::Tensor dropMaskAt = at::Tensor();
    if (pDropout != 0) {
        // int64_t length = (input.numel() + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        // dropMaskAt = at_npu::native::OpPreparation::apply_tensor_without_format({length}, input.options().dtype(at::kByte));
    }

    at::Tensor attentionMaskAt = at::Tensor();
    if (isCausal) {
        // NOTE: reference to: https://gitee.com/ascend/ModelLink/blob/v0.1.0/pretrain_llama.py#L74
        // It should be noted that the generation logic of attention mask is exactly opposite to common sense on ascend.
        attentionMaskAt = npu_preparation::apply_tensor_without_format({s0, s1}, qAt.options().dtype(at::kBool));  // [S0, S1]
        EXEC_NPU_CMD(aclnnInplaceOne, attentionMaskAt);
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskAt, diagonal);
    }

    int64_t preTokens = kAt.size(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;

    at::Tensor softmaxMaxAt;
    at::Tensor softmaxSumAt;
    at::Tensor softmaxOutAt;

    softmaxMaxAt = at_npu::native::OpPreparation::apply_tensor_without_format({b, n, s0, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [B, N, S0, 8]
    softmaxSumAt = at_npu::native::OpPreparation::apply_tensor_without_format({b, n, s0, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [B, N, S0, 8]
    softmaxOutAt = at::empty({0}, qAt.options());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScore,
                                 qAt,
                                 kAt,
                                 vAt,
                                 pseAt,
                                 dropMaskAt,
                                 paddingMaskAt,
                                 attentionMaskAt,
                                 prefixN,
                                 softmaxScale,
                                 keepProb,
                                 preTokens,
                                 nextTokens,
                                 n,
                                 inputLayoutPtr,
                                 innerPrecise,
                                 sparseMode,
                                 softmaxMaxAt,
                                 softmaxSumAt,
                                 softmaxOutAt,
                                 attentionOutAt);

    impl::aten::buildDiopiTensor(ctx, softmaxMaxAt, softmaxMax);
    impl::aten::buildDiopiTensor(ctx, softmaxSumAt, softmaxSum);
    impl::aten::buildDiopiTensor(ctx, softmaxOutAt, softmaxOut);
    END_CALL_ACL_OP();
}

diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                         diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                         diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                         diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal) {
    BEGIN_CALL_ACL_OP(q, k, v, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, gradQ, gradK, gradV, gradOut);

    DIOPI_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4-dimensional");
    DIOPI_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4-dimensional");
    DIOPI_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    std::string inputLayout = "BSND";
    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());

    int64_t b = qAt.size(0);
    int64_t s0 = qAt.size(1);  // S for query
    int64_t s1 = kAt.size(1);  // S for key & value
    int64_t n = qAt.size(2);
    int64_t d = qAt.size(3);

    double keepProb = 1 - pDropout;

    at::Tensor pseAt = at::Tensor();
    at::Tensor gradPseAt = at::empty({0}, qAt.options());
    at::IntArrayRef prefixN = at::IntArrayRef{};

    at::Tensor paddingMaskAt = at::Tensor();

    at::Tensor dropMaskAt = at::Tensor();
    if (pDropout != 0) {
        // int64_t length = (input.numel() + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        // dropMaskAt = at_npu::native::OpPreparation::apply_tensor_without_format({length}, input.options().dtype(at::kByte));
    }

    at::Tensor attentionMaskAt = at::Tensor();
    if (isCausal) {
    }

    int64_t preTokens = kAt.size(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScoreGrad,
                                 qAt,
                                 kAt,
                                 vAt,
                                 gradOutAt,
                                 pseAt,
                                 dropMaskAt,
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
                                 n,
                                 inputLayoutPtr,
                                 innerPrecise,
                                 sparseMode,
                                 gradQAt,
                                 gradKAt,
                                 gradVAt,
                                 gradPseAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
