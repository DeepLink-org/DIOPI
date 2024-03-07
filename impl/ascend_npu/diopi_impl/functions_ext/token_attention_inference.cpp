/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    BEGIN_CALL_ACL_OP(attentionOut, q, k, bLoc, bStartLoc, bSeqLen);
    int batch = bLocAt.size(0);
    int head = qAt.size(1);
    int dim = qAt.size(2);
    qAt = qAt.reshape({batch, 1, head, dim}).transpose(1, 2);
    c10::ScalarType dtype = qAt.scalar_type();
    c10::Device device = qAt.device();
    c10::Layout layout = qAt.layout();
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = bSeqLenAt[i].item<int>();
        int curSeqStartLoc = bStartLocAt[i].item<int>();
        at::Tensor kLoc = op_api::index_select(bLocAt[i], 0, op_api::arange(maxInputLen - curSeqLen, maxInputLen, at::kInt, layout, device));
        at::Tensor key = op_api::index(kAt, {kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
        at::Tensor outLoc = op_api::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen, at::kInt, layout, device);
        at::Tensor values =
            (op_api::matmul(op_api::index(qAt, {torch::scalar_to_tensor(i)}).toType(at::kFloat), key.transpose(2, 3).toType(at::kFloat)) / std::sqrt(dim))
                .view({head, curSeqLen})
                .toType(dtype);
        op_api::index_put_(attentionOutAt, {at::Tensor(), outLoc}, values);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
