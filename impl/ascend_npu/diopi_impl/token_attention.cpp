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
    std::cout << "batch=" << batch << std::endl;
    int head = qAt.size(1);
    std::cout << "head=" << head << std::endl;
    int dim = qAt.size(2);
    std::cout << "dim=" << dim << std::endl;

    qAt = acl_op::npu_reshape(qAt, {batch, 1, head, dim});
    std::cout << "qAt reshape finish. " << ", batch=" << batch << ", head=" << head << ", dim=" << dim << std::endl;
    qAt = acl_op::npu_transpose(qAt, {0, 2, 1, 3});
    std::cout << "qAt transpose finish." << ", batch=" << batch << ", head=" << head << ", dim=" << dim << std::endl;

    std:: cout << "====== batch=" << batch << ", head=" << head << ", dim=" << dim << std::endl;
    // qAt = qAt.reshape({batch, 1, head, dim}).transpose(1, 2);
    // std::cout << "qAt = qAt.reshape({batch, 1, head, dim}).transpose(1, 2); finish.\n";
    for (int i = 0; i < batch; ++i) {
        std::cout << "come into for loop" << std::endl;
        int curSeqLen = bSeqLenAt[i].item<int>();
        int curSeqStartLoc = bStartLocAt[i].item<int>();
        std::cout << "curSeqLen=" << curSeqLen << ", curSeqStartLoc=" << curSeqStartLoc << std::endl;
        at::Tensor kLoc = acl_op::index_select(bLocAt[i], 0, acl_op::arange(at::Scalar(maxInputLen - curSeqLen), at::Scalar(maxInputLen), at::Scalar(1), at::ScalarType::Int));
        at::Tensor key = acl_op::index(kAt, {kLoc});
        key = impl::aten::view(key, {1, curSeqLen, head, dim});
        key = acl_op::npu_transpose(key, {0, 2, 1, 3});
        at::Tensor outLoc = acl_op::arange(at::Scalar(curSeqStartLoc), at::Scalar(curSeqStartLoc + curSeqLen), at::Scalar(1), at::ScalarType::Int);
        // auto index = acl_op::index(qAt, {i});
        auto keyTrans = acl_op::npu_transpose(key, {0, 1, 3, 2});
        at::Tensor valuesTmp = (acl_op::matmul(qAt.index({i}), keyTrans) / std::sqrt(dim));
        // .view({head, curSeqLen});
        at::Tensor values = impl::aten::view(valuesTmp, {head, curSeqLen});

        // attentionOutAt.index_put_({torch::indexing::Slice(), outLoc}, values);
        at::Tensor attentionSlice = custom_slice(attentionOutAt, outLoc);
        attentionSlice.copy_(values.view({-1}));
    }
    END_CALL_ACL_OP();
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
