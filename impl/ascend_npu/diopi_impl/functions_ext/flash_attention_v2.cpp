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

}  // namespace

diopiError_t diopiFlashAttentionV2(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* dropoutMask,
                                   diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum, diopiTensorHandle_t* softmaxOut,
                                   diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                   diopiConstTensorHandle_t alibiSlopes, diopiConstTensorHandle_t attentionMask, float pDropout, float softmaxScale,
                                   bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    BEGIN_CALL_ACL_OP(q, k, v, alibiSlopes, gen, attentionOut);

    DIOPI_CHECK(alibiSlopes == nullptr, "For ascend, flash attention currently does not support Attention with Linear Biases (ALiBi)!");
    DIOPI_CHECK(windowSizeLeft == -1 && windowSizeRight == -1, "For ascend, flash attention currently does not support sliding window local attention!");
    DIOPI_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4-dimensional");
    DIOPI_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4-dimensional");
    DIOPI_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    int64_t b = qAt.size(0);
    int64_t s0 = qAt.size(1);  // S for query
    int64_t s1 = kAt.size(1);  // S for key & value
    int64_t n = qAt.size(2);
    int64_t d = qAt.size(3);
    const char* inputLayout = "BSND";

    double keepProb = static_cast<double>(1 - pDropout);

    at::Tensor pseAt = at::Tensor();
    at::IntArrayRef prefixN = at::IntArrayRef{};

    at::Tensor paddingMaskAt = at::Tensor();

    at::Tensor dropoutMaskAt = at::Tensor();
    if (pDropout > 0 && pDropout <= 1) {
        int64_t numels = b * n * s0 * s1;  // [B,N,S,S]
        int64_t length = (numels + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        dropoutMaskAt = npu_preparation::apply_tensor_without_format({length}, qAt.options().dtype(at::kByte));
        if (pDropout == 1) {
            EXEC_NPU_CMD(aclnnInplaceZero, dropoutMaskAt);
        } else {
            std::vector<int64_t> shapeVector{numels};
            at::IntArrayRef shapeArray(shapeVector);
            auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(genAt)->philox_engine_inputs(10);
            const uint64_t seed = pair.first;
            const uint64_t offset = pair.second;
            EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, pDropout, seed, offset, dropoutMaskAt);
        }
    }

    // In order to reduce overhead, a common attention mask can be saved in the model layer, which can avoid repeatedly generating the attention mask every time
    // the flash attention is calculated.
    // It is worth noting that the attention mask used by ascend is contrary to common sense, so this interface cannot be used as a universal diopi interface
    // for the time being. Please do not use it with other devices.
    at::Tensor attentionMaskAt = at::Tensor();
    int64_t sparseMode = 0;
    if (isCausal) {
        // NOTE: reference to: https://gitee.com/ascend/ModelLink/blob/v0.1.0/pretrain_llama.py#L74
        // It should be noted that the generation logic of attention mask is exactly opposite to common sense on ascend.
        // According to Huawei documentation, when the attentionMask shape is greater than 2048 * 2048, sparseMode=2 can be adjusted to reduce the memory usage:
        // https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000742.html
        if (s0 > 2048 && s1 > 2048) {
            attentionMaskAt = npu_preparation::apply_tensor_without_format({2048, 2048}, qAt.options().dtype(at::kBool));  // [2048, 2048]
            sparseMode = 2;
        } else {
            attentionMaskAt = npu_preparation::apply_tensor_without_format({s0, s1}, qAt.options().dtype(at::kBool));  // [S0, S1]
        }
        EXEC_NPU_CMD(aclnnInplaceOne, attentionMaskAt);
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskAt, diagonal);
    }

    int64_t preTokens = kAt.size(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;

    at::Tensor softmaxMaxAt;
    at::Tensor softmaxSumAt;
    at::Tensor softmaxOutAt;

    softmaxMaxAt = at_npu::native::OpPreparation::apply_tensor_without_format({b, n, s0, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [B, N, S0, 8]
    softmaxSumAt = at_npu::native::OpPreparation::apply_tensor_without_format({b, n, s0, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [B, N, S0, 8]
    softmaxOutAt = at_npu::native::OpPreparation::apply_tensor_without_format({0},
                                                                              qAt.options());  // [0]
    double scale = static_cast<double>(softmaxScale);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionScore,
                                 qAt,
                                 kAt,
                                 vAt,
                                 pseAt,
                                 dropoutMaskAt,
                                 paddingMaskAt,
                                 attentionMaskAt,
                                 prefixN,
                                 scale,
                                 keepProb,
                                 preTokens,
                                 nextTokens,
                                 n,
                                 inputLayout,
                                 innerPrecise,
                                 sparseMode,
                                 softmaxMaxAt,
                                 softmaxSumAt,
                                 softmaxOutAt,
                                 attentionOutAt);

    if (dropoutMaskAt.defined()) {
        impl::aten::buildDiopiTensor(ctx, dropoutMaskAt, dropoutMask);
    }
    impl::aten::buildDiopiTensor(ctx, softmaxMaxAt, softmaxMax);
    impl::aten::buildDiopiTensor(ctx, softmaxSumAt, softmaxSum);
    impl::aten::buildDiopiTensor(ctx, softmaxOutAt, softmaxOut);
    END_CALL_ACL_OP();
}

diopiError_t diopiFlashAttentionV2Backward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                           diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                           diopiConstTensorHandle_t alibiSlopes, diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t attentionMask,
                                           diopiConstTensorHandle_t dropoutMask, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                           diopiConstTensorHandle_t softmaxOut, float pDropout, float softmaxScale, bool isCausal, int32_t windowSizeLeft,
                                           int32_t windowSizeRight) {
    BEGIN_CALL_ACL_OP(q, k, v, attentionOut, softmaxMax, softmaxSum, softmaxOut, gradQ, gradK, gradV, gradOut);

    DIOPI_CHECK(alibiSlopes == nullptr, "For ascend, flash attention currently does not support Attention with Linear Biases (ALiBi)!");
    DIOPI_CHECK(windowSizeLeft == -1 && windowSizeRight == -1, "For ascend, flash attention currently does not support sliding window local attention!");
    DIOPI_CHECK(qAt.dim() == 4, "The shapes of the input query should be 4-dimensional");
    DIOPI_CHECK(kAt.dim() == 4, "The shapes of the input key should be 4-dimensional");
    DIOPI_CHECK(vAt.dim() == 4, "The shapes of the input value should be 4-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    at::Tensor dropoutMaskAt;
    if (dropoutMask) {
        dropoutMaskAt = impl::aten::buildATen(dropoutMask);
    }

    int64_t b = qAt.size(0);
    int64_t s0 = qAt.size(1);  // S for query
    int64_t s1 = kAt.size(1);  // S for key & value
    int64_t n = qAt.size(2);
    int64_t d = qAt.size(3);
    const char* inputLayout = "BSND";

    double keepProb = static_cast<double>(1 - pDropout);

    at::Tensor pseAt = at::Tensor();
    at::Tensor gradPseAt = at_npu::native::OpPreparation::apply_tensor_without_format({0}, qAt.options());
    at::IntArrayRef prefixN = at::IntArrayRef{};

    at::Tensor paddingMaskAt = at::Tensor();

    int64_t preTokens = kAt.size(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;

    // In order to reduce overhead, a common attention mask can be saved in the model layer, which can avoid repeatedly generating the attention mask every time
    // the flash attention is calculated.
    // It is worth noting that the attention mask used by ascend is contrary to common sense, so this interface cannot be used as a universal diopi interface
    // for the time being. Please do not use it with other devices.
    at::Tensor attentionMaskAt = at::Tensor();
    int64_t sparseMode = 0;
    if (isCausal) {
        // NOTE: reference to: https://gitee.com/ascend/ModelLink/blob/v0.1.0/pretrain_llama.py#L74
        // It should be noted that the generation logic of attention mask is exactly opposite to common sense on ascend.
        // According to Huawei documentation, when the attentionMask shape is greater than 2048 * 2048, sparseMode=2 can be adjusted to reduce the memory usage:
        // https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000742.html
        if (s0 > 2048 && s1 > 2048) {
            attentionMaskAt = npu_preparation::apply_tensor_without_format({2048, 2048}, qAt.options().dtype(at::kBool));  // [2048, 2048]
            sparseMode = 2;
        } else {
            attentionMaskAt = npu_preparation::apply_tensor_without_format({s0, s1}, qAt.options().dtype(at::kBool));  // [S0, S1]
        }
        EXEC_NPU_CMD(aclnnInplaceOne, attentionMaskAt);
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskAt, diagonal);
    }

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
                                 n,
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
