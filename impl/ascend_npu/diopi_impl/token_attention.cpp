/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

at::Tensor custom_slice(const at::Tensor& tensor, at::Tensor indices) {
    at::Tensor result = at::empty({indices.size(0)}, tensor.options());
    for (int i = 0; i < indices.size(0); ++i) {
        result[i] = tensor[indices[i].item<int>()];
    }
    return result;
}

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    BEGIN_CALL_ACL_OP(attentionOut, q, k, bLoc, bStartLoc, bSeqLen);
    int batch = bLocAt.size(0);
    int head = qAt.size(1);
    int dim = qAt.size(2);

    qAt = qAt.reshape({batch, 1, head, dim}).transpose(1, 2);
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = bSeqLenAt[i].item<int>();
        int curSeqStartLoc = bStartLocAt[i].item<int>();
        auto start1 = at::Scalar(maxInputLen - curSeqLen);
        auto end1 = at::Scalar(maxInputLen);
        at::Tensor kLoc = bLocAt[i].index_select(0, acl_op::arange(at::Scalar(maxInputLen - curSeqLen), at::Scalar(maxInputLen), at::Scalar(1), at::ScalarType::Int));
        at::Tensor key = kAt.index({kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
        auto start2 = at::Scalar(curSeqStartLoc);
        auto end2 = at::Scalar(curSeqStartLoc + curSeqLen);
        at::Tensor outLoc = acl_op::arange(at::Scalar(curSeqStartLoc), at::Scalar(curSeqStartLoc + curSeqLen), at::Scalar(1), at::ScalarType::Int);
        at::Tensor values = (acl_op::matmul(qAt.index({i}), key.transpose(2, 3)) / std::sqrt(dim)).view({head, curSeqLen});

        // attentionOutAt.index_put_({torch::indexing::Slice(), outLoc}, values);
        at::Tensor attentionSlice = custom_slice(attentionOutAt, outLoc);
        attentionSlice.copy_(values.view({-1}));
    }
    END_CALL_ACL_OP();
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
