/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <torch/torch.h>

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiTokenAttentionInference(
    diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut,
    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
    diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc,
    diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
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
    at::Tensor kLoc = at::index_select(
        atBLoc[i], 0,
        at::arange(maxInputLen - curSeqLen, maxInputLen).to(atBLoc.device()));
    at::Tensor key =
        at::index(atK, {kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
    at::Tensor outLoc = at::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen)
                            .to(atBLoc.device());
    auto a = at::index(atQ, {torch::scalar_to_tensor(i)});
    auto b = key.transpose(2, 3);
    auto mat = acl_op::matmul(a.to(at::ScalarType::Float),
                              b.to(at::ScalarType::Float));
    at::Tensor values = (mat / std::sqrt(dim)).view({head, curSeqLen});

    at::index_put_(attentionOutAt, {at::Tensor(), outLoc}, values);
  }

  END_CALL_ACL_OP();
}

diopiError_t diopiTokenSoftmaxReduceVInference(
    diopiContextHandle_t ctx, diopiTensorHandle_t out,
    diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
    diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc,
    diopiConstTensorHandle_t bSeqLen, int maxInputLen, int otherKVIndex) {
  impl::aten::setCurCtx(ctx);
  at::Tensor atOut = impl::aten::buildATen(out);
  at::Tensor atV = impl::aten::buildATen(v);
  at::Tensor atLogics = impl::aten::buildATen(logics);
  at::Tensor atBLoc = impl::aten::buildATen(bLoc);
  at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
  at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);

  int batch = atBLoc.size(0);
  int head = atV.size(1);
  int dim = atV.size(2);

  for (int i = 0; i < batch; ++i) {
    int curSeqLen = atBSeqLen[i].item<int>();
    int curSeqStartLoc = atBStartLoc[i].item<int>();
    at::Tensor P = atLogics.slice(1, curSeqStartLoc, curSeqStartLoc + curSeqLen)
                       .softmax(-1)
                       .reshape({head, 1, 1, curSeqLen})
                       .transpose(0, 1)
                       .to(atLogics.device());
    at::Tensor vLoc = atBLoc[i].index_select(
        0,
        at::arange(maxInputLen - curSeqLen, maxInputLen).to(atLogics.device()));
    at::Tensor V =
        atV.index({vLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
    atOut[i] = at::matmul(P, V).view({head, dim});
  }
  impl::aten::unsetCurCtx();
  return diopiSuccess;
}

} // namespace OP_IMPL_NS
