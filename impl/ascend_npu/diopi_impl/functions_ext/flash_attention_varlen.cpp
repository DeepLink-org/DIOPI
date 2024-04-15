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

diopiError_t diopiFlashAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* attentionMask,
                                       diopiTensorHandle_t* dropoutMask, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                                       diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                       diopiConstTensorHandle_t v, diopiConstTensorHandle_t cumSeqQ, diopiConstTensorHandle_t cumSeqK, double pDropout,
                                       double softmaxScale, bool isCausal) {
    BEGIN_CALL_ACL_OP(q, k, v, cumSeqQ, cumSeqK, gen, attentionOut);

    DIOPI_CHECK(qAt.dim() == 3, "The shapes of the input query should be 3-dimensional");
    DIOPI_CHECK(kAt.dim() == 3, "The shapes of the input key should be 3-dimensional");
    DIOPI_CHECK(vAt.dim() == 3, "The shapes of the input value should be 3-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    std::string inputLayout = "TND";
    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());

    int64_t t = qAt.size(0);
    int64_t n = qAt.size(1);
    int64_t d = qAt.size(2);

    double keepProb = 1 - pDropout;

    at::Tensor pseAt = at::Tensor();
    at::IntArrayRef prefixN = at::IntArrayRef{};

    at::Tensor paddingMaskAt = at::Tensor();

    at::Tensor dropoutMaskAt = at::Tensor();
    if (pDropout > 0 && pDropout <= 1) {
        int64_t numels = n;
        int64_t accum = cumSeqQAt[0].item<int64_t>() * cumSeqKAt[0].item<int64_t>();
        for (int64_t i = 1; i < cumSeqQAt.dim(); i++) {
            accum += ((cumSeqQAt[i].item<int64_t>() - cumSeqQAt[i - 1].item<int64_t>()) * (cumSeqKAt[i].item<int64_t>() - cumSeqKAt[i - 1].item<int64_t>()));
        }
        numels *= accum;
        int64_t length = (numels + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
        dropoutMaskAt = npu_preparation::apply_tensor_without_format({length}, qAt.options().dtype(at::kByte));
        if (pDropout == 1) {
            op_api::zero_(dropoutMaskAt);
        } else {
            at::IntArrayRef shapeArray(numels);
            auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(genAt)->philox_engine_inputs(10);
            const uint64_t seed = pair.first;
            const uint64_t offset = pair.second;
            EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, pDropout, seed, offset, dropoutMaskAt);
        }
    }

    at::Tensor attentionMaskAt = at::Tensor();
    if (isCausal) {
        attentionMaskAt = npu_preparation::apply_tensor_without_format({t, t}, qAt.options().dtype(at::kBool));
        EXEC_NPU_CMD(aclnnInplaceOne, attentionMaskAt);
        int64_t diagonal = 1;
        EXEC_NPU_CMD(aclnnInplaceTriu, attentionMaskAt, diagonal);
    }

    int64_t preTokens = kAt.size(0);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;

    at::Tensor softmaxMaxAt;
    at::Tensor softmaxSumAt;
    at::Tensor softmaxOutAt;

    softmaxMaxAt = at_npu::native::OpPreparation::apply_tensor_without_format({t, n, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [T, N, 8]
    softmaxSumAt = at_npu::native::OpPreparation::apply_tensor_without_format({t, n, 8},
                                                                              qAt.options().dtype(at::kFloat));  // [T, N, 8]
    softmaxOutAt = at_npu::native::OpPreparation::apply_tensor_without_format({0},
                                                                              qAt.options());  // [0]

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionVarLenScore,
                                 qAt,
                                 kAt,
                                 vAt,
                                 pseAt,
                                 dropoutMaskAt,
                                 paddingMaskAt,
                                 attentionMaskAt,
                                 prefixN,
                                 cumSeqQAt,
                                 cumSeqKAt,
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

    if (attentionMaskAt.defined()) {
        impl::aten::buildDiopiTensor(ctx, attentionMaskAt, attentionMask);
    }
    if (dropoutMaskAt.defined()) {
        impl::aten::buildDiopiTensor(ctx, dropoutMaskAt, dropoutMask);
    }
    impl::aten::buildDiopiTensor(ctx, softmaxMaxAt, softmaxMax);
    impl::aten::buildDiopiTensor(ctx, softmaxSumAt, softmaxSum);
    impl::aten::buildDiopiTensor(ctx, softmaxOutAt, softmaxOut);
    END_CALL_ACL_OP();
}

diopiError_t diopiFlashAttentionVarLenBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                               diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                               diopiConstTensorHandle_t v, diopiConstTensorHandle_t cumSeqQ, diopiConstTensorHandle_t cumSeqK,
                                               diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t attentionMask,
                                               diopiConstTensorHandle_t dropoutMask, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                               diopiConstTensorHandle_t softmaxOut, double pDropout, double softmaxScale) {
    BEGIN_CALL_ACL_OP(q, k, v, cumSeqQ, cumSeqK, attentionOut, softmaxMax, softmaxSum, softmaxOut, gradQ, gradK, gradV, gradOut);

    at::Tensor dropoutMaskAt;
    at::Tensor attentionMaskAt;
    if (dropoutMask) {
        dropoutMaskAt = impl::aten::buildATen(dropoutMask);
    }
    if (attentionMask) {
        attentionMaskAt = impl::aten::buildATen(attentionMask);
    }

    DIOPI_CHECK(qAt.dim() == 3, "The shapes of the input query should be 3-dimensional");
    DIOPI_CHECK(kAt.dim() == 3, "The shapes of the input key should be 3-dimensional");
    DIOPI_CHECK(vAt.dim() == 3, "The shapes of the input value should be 3-dimensional");
    DIOPI_CHECK(pDropout >= 0 && pDropout <= 1, "The p_dropout value must be in range of [0, 1]");

    std::string inputLayout = "TND";
    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());

    int64_t t = qAt.size(0);
    int64_t n = qAt.size(1);
    int64_t d = qAt.size(2);

    double keepProb = 1 - pDropout;

    at::Tensor pseAt = at::Tensor();
    at::Tensor gradPseAt = at_npu::native::OpPreparation::apply_tensor_without_format({0}, qAt.options());
    at::IntArrayRef prefixN = at::IntArrayRef{};

    at::Tensor paddingMaskAt = at::Tensor();

    int64_t preTokens = kAt.size(0);
    int64_t nextTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFlashAttentionUnpaddingScoreGrad,
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
                                 cumSeqQAt,
                                 cumSeqKAt,
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
