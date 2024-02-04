/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "../helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                               int maxInputLen, int otherKVIndex) {
    BEGIN_CALL_ACL_OP(out, logics, v, bLoc, bStartLoc, bSeqLen);
    int batch = bLocAt.size(0);
    int head = vAt.size(1);
    int dim = vAt.size(2);
    c10::ScalarType dtype = logicsAt.scalar_type();
    c10::Device device = logicsAt.device();
    c10::Layout layout = logicsAt.layout();
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = bSeqLenAt[i].item<int>();
        int curSeqStartLoc = bStartLocAt[i].item<int>();
        at::Tensor p = at::index(logicsAt, {at::Tensor(), acl_op::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen, at::kInt, layout, device)})
                           .softmax(-1)
                           .reshape({head, 1, 1, curSeqLen})
                           .transpose(0, 1);
        at::Tensor vLoc = bLocAt[i].index_select(0, acl_op::arange(maxInputLen - curSeqLen, maxInputLen, at::kInt, layout, device));
        at::Tensor v = at::index(vAt, {vLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
        at::Tensor values = at::matmul(p.toType(at::kFloat), v.toType(at::kFloat)).view({head, dim}).toType(dtype);
        at::index_put_(outAt, {torch::scalar_to_tensor(i)}, values);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
