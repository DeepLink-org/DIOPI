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
    int64_t B = qAt.size(0);
    int64_t S0 = qAt.size(1);  // S for query
    int64_t S1 = kAt.size(1);  // S for key & value
    int64_t N = qAt.size(2);
    int64_t D = qAt.size(3);
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
    }

    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());
    int64_t preTokens = kAt.size(1);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;  // 保留参数，暂未使用。0, fp16 high precision. 1, high performance.
    int64_t sparseMode = 0;

    at::Tensor softmaxMaxAt;
    at::Tensor softmaxSumAt;
    at::Tensor softmaxOutAt;

    softmaxMaxAt = at_npu::native::OpPreparation::apply_tensor_without_format({B, N, S0, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [B, N, S0, 8]
    softmaxSumAt = at_npu::native::OpPreparation::apply_tensor_without_format({B, N, S0, 8},
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
                                 N,
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

}  // namespace OP_IMPL_NS
