/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    BEGIN_CALL_ACL_OP(attentionOut);
    at::Tensor atQ = impl::aten::buildATen(q);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atBLoc = impl::aten::buildATen(bLoc);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);

    int batch = atBLoc.size(0);
    int head = atQ.size(1);
    int dim = atQ.size(2);
    atQ = atQ.reshape({batch, 1, head, dim}).transpose(1, 2);
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = atBSeqLen[i].item<int>();
        int curSeqStartLoc = atBStartLoc[i].item<int>();
        at::Tensor kLoc = at::index_select(atBLoc[i], 0, at::arange(maxInputLen - curSeqLen, maxInputLen).to(atBLoc.device()));
        at::Tensor key = at::index(atK, {kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
        at::Tensor outLoc = at::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen).to(atBLoc.device());
        auto a = at::index(atQ, {torch::scalar_to_tensor(i)});
        auto b = key.transpose(2, 3);
        auto mat = acl_op::matmul(a.to(at::ScalarType::Float), b.to(at::ScalarType::Float));
        at::Tensor values = (mat / std::sqrt(dim)).view({head, curSeqLen});

        at::index_put_(attentionOutAt, {at::Tensor(), outLoc}, values);
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
