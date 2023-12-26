/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

at::Tensor torchContextAttention(at::Tensor xq, at::Tensor xk, at::Tensor xv, int batchSize, int seqLen, int head, int dim) {
    c10::ScalarType dtype = xq.scalar_type();
    c10::Device device = xq.device();
    c10::Layout layout = xq.layout();
    xq = xq.view({batchSize, seqLen, head, dim}).transpose(1, 2);
    xk = xk.view({batchSize, seqLen, head, dim}).transpose(1, 2);
    xv = xv.view({batchSize, seqLen, head, dim}).transpose(1, 2);
    at::Tensor mask = acl_op::tril(acl_op::ones({seqLen, seqLen}, at::kFloat, layout, device)).unsqueeze(0).unsqueeze(0);
    mask.masked_fill_(mask == 0., -100000000.0);
    mask = mask.repeat({batchSize, head, 1, 1});
    at::Tensor scores = at::matmul(xq.to(at::kFloat), xk.transpose(2, 3).to(at::kFloat)) / std::sqrt(dim);
    at::Tensor output = at::matmul((scores + mask).softmax(-1), xv.to(at::kFloat)).transpose(1, 2).to(dtype);
    output = output.view({output.numel() / static_cast<int64_t>(head * dim), head, dim});
    return output;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
    BEGIN_CALL_ACL_OP(out, q, k, v, bStartLoc, bSeqLen);
    int batch = bStartLocAt.size(0);
    int head = qAt.size(1);
    int dim = qAt.size(2);
    c10::Device device = qAt.device();
    c10::Layout layout = qAt.layout();
    for (int i = 0; i < batch; ++i) {
        int start = bStartLocAt[i].item<int>();
        int end = start + bSeqLenAt[i].item<int>();
        at::Tensor slice = acl_op::arange(start, end, at::kLong, layout, device);
        at::Tensor values =
            torchContextAttention(at::index(qAt, {slice}), at::index(kAt, {slice}), at::index(vAt, {slice}), 1, bSeqLenAt[i].item<int>(), head, dim);
        at::index_put_(outAt, {slice}, values);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
