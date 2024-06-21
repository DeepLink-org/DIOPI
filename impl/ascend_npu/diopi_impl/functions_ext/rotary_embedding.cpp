/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

at::Tensor viewAs4D(const at::Tensor& input) {
    if (input.dim() == 4) {
        return input;
    }

    static const int64_t n = 4;
    int dim = input.dim();
    std::vector<int64_t> viewShape(n, 1);
    auto inputShape = input.sizes();
    for (int i = 0; i < dim; ++i) {
        viewShape[i + n - dim] = inputShape[i];
    }
    return impl::aten::viewStorage(input, viewShape);
}

DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
    if (interleaved) {
        set_last_error_string("interleaved rotary embedding is not supported yet");
        return diopiNoImplement;
    }

    BEGIN_CALL_ACL_OP(out, x, cos, sin);
    TORCH_CHECK(xAt.size(-1) == 2 * cosAt.size(-1) && xAt.size(-1) == 2 * sinAt.size(-1),
                "The size of the last dimension of x must be twice the size of the corresponding dimensions of cos and sin!");
    if (xAt.numel() == 0) {
        END_CALL_ACL_OP();
    }

    if (xAt.dim() >= 5) {
        set_last_error_string("rotary embedding not support 5D tensor yet");
        return diopi5DNotSupported;
    }

    at::Tensor xView = viewAs4D(xAt);
    at::Tensor outView = viewAs4D(outAt);
    at::Tensor cosView = viewAs4D(cosAt);
    at::Tensor sinView = viewAs4D(sinAt);
    // To meet the ascend kernel requirement: the last dimension size of cos and sin is the same as the dimension size corresponding to input, use cat op to
    // concatenate in the last dimension.
    at::Tensor cosCat = op_api::cat({cosView, cosView}, -1);
    at::Tensor sinCat = op_api::cat({sinView, sinView}, -1);
    if (conj) {
        op_api::neg_(sinCat);
    }

    // According to API document
    // https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/foundmodeltrain/PT_LMTMOG_000111.html,
    // the last dimension should be divisible by 128 when using RotaryMul operator.
    // If input dtype is fp16, testcase will pass when the last dimension is divisible by 32.
    // In order to achieve good performance on internlm, we use 32 and skip some testcases.

    std::vector<at::Tensor> chunkResult = xView.chunk(2, -1);
    at::Tensor xNew = op_api::cat({chunkResult[1] * (-1), chunkResult[0]}, -1);
    auto result1 = op_api::mul(cosCat, xView);
    auto result2 = op_api::mul(sinCat, xNew);
    op_api::add_out(result1, result2, 1.0, outView);

    END_CALL_ACL_OP();
}

DIOPI_API diopiError_t diopiRotaryEmbeddingV2(diopiContextHandle_t ctx, diopiTensorHandle_t query, diopiTensorHandle_t key, diopiConstTensorHandle_t cos,
                                              diopiConstTensorHandle_t sin, int64_t dim) {
    BEGIN_CALL_ACL_OP(query, key, cos, sin);
    int64_t layOut = 1;
    if (queryAt.dim() == 2) {
        queryAt = impl::aten::viewStorage(queryAt, {1, queryAt.size(0), queryAt.size(1) / dim, dim});
    }
    if (keyAt.dim() == 2) {
        keyAt = impl::aten::viewStorage(keyAt, {1, keyAt.size(0), keyAt.size(1) / dim, dim});
    }
    cosAt = viewAs4D(cosAt);
    sinAt = viewAs4D(sinAt);
    EXEC_NPU_CMD(aclnnApplyRotaryPosEmb, queryAt, keyAt, cosAt, sinAt, layOut);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
