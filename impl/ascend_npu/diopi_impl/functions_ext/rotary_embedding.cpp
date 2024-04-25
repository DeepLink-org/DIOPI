/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

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
    if (xAt.numel() == 0) {
        END_CALL_ACL_OP();
    }

    if (xAt.dim() >= 5) {
        set_last_error_string("rotary embedding not support 5D tensor yet");
        impl::aten::unsetCurCtx();
        return diopi5DNotSupported;
    }

    at::Tensor xView = viewAs4D(xAt);
    at::Tensor outView = viewAs4D(outAt);
    at::Tensor cosView = viewAs4D(cosAt);
    at::Tensor sinView = viewAs4D(sinAt);
    at::Tensor cosRepeated = cosView;
    at::Tensor sinRepeated = sinView;
    TORCH_CHECK((cosRepeated.sizes()[3] * 2 == xView.sizes()[3] || cosRepeated.sizes()[3] == xView.sizes()[3]),
                "thd -1 dim of cos|sin must be half or same of input's");
    if (cosRepeated.sizes()[3] * 2 == xView.sizes()[3]) {
        cosRepeated = op_api::repeat(cosView, {1, 1, 1, 2});
        sinRepeated = op_api::repeat(sinView, {1, 1, 1, 2});
    }
    if (conj) {
        op_api::neg_(sinRepeated);
    }

    // According to API document
    // https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/foundmodeltrain/PT_LMTMOG_000111.html,
    // the last dimension should be divisible by 128 when using RotaryMul operator.
    // If input dtype is fp16, testcase will pass when the last dimension is divisible by 32.
    // In order to achieve good performance on internlm, we use 32 and skip some testcases.

    std::vector<at::Tensor> chunkResult = xView.chunk(2, -1);
    at::Tensor xNew = op_api::cat({chunkResult[1] * (-1), chunkResult[0]}, -1);
    at::Tensor result = op_api::mul(cosRepeated, xView) + op_api::mul(sinRepeated, xNew);
    outView.copy_(result);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
