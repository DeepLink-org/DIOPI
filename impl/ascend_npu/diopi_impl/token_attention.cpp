/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

namespace OP_IMPL_NS {

void printTensor(at::Tensor &b) {
  std::cout << "b.device()=" << b.device() << std::endl;
  std::cout << "b.sizes()=" << b.sizes() << std::endl;
  std::cout << "b.dtype()=" << b.dtype() << std::endl;
}

diopiError_t diopiTokenAttentionInference(
    diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut,
    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
    diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc,
    diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
  BEGIN_CALL_ACL_OP(attentionOut);
  // impl::aten::setCurCtx(ctx);
  at::Tensor atQ = impl::aten::buildATen(q);
  at::Tensor atK = impl::aten::buildATen(k);
  at::Tensor atBLoc = impl::aten::buildATen(bLoc);
  at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
  at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);
  at::Tensor atAttentionOut = impl::aten::buildATen(attentionOut);
  atAttentionOut = atAttentionOut.cpu().to(at::ScalarType::Float);

  // atQ = atQ.cpu();
  // atK = atK.cpu();
  // atBLoc = atBLoc.cpu();
  // atBStartLoc = atBStartLoc.cpu();
  // atBSeqLen = atBSeqLen.cpu();
  caffe2::TypeMeta dtype = atAttentionOut.dtype();
  // atAttentionOut = atAttentionOut.cpu().to(at::ScalarType::Float);

  int batch = atBLoc.size(0);
  int head = atQ.size(1);
  int dim = atQ.size(2);

  atQ = atQ.reshape({batch, 1, head, dim}).transpose(1, 2);
  std::cout << "come into loop." << std::endl;
  for (int i = 0; i < batch; ++i) {
    int curSeqLen = atBSeqLen[i].item<int>();
    int curSeqStartLoc = atBStartLoc[i].item<int>();
    at::Tensor kLoc = atBLoc[i].index_select(
        0, at::arange(maxInputLen - curSeqLen, maxInputLen));
    std::cout << "kLoc=" << kLoc.sizes() << std::endl;
    // at::Tensor key =
    //     atK.index({kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
    at::Tensor key =
        at::index(atK, {kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
    std::cout << "key=" << key.sizes() << std::endl;
    at::Tensor outLoc = at::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen);
    auto iScalar = torch::scalar_to_tensor(i);
    auto a = at::index(atQ, {iScalar});
    auto b = key.transpose(2, 3);
    std::cout << "a.device()=" << a.device() << std::endl;
    std::cout << "b.device()=" << b.device() << std::endl;
    at::Tensor values =
        (acl_op::matmul(a.to(at::ScalarType::Long), b.to(at::ScalarType::Long)) / std::sqrt(dim)).view({head, curSeqLen});
    atAttentionOut.index_put_({torch::indexing::Slice(), outLoc}, values.cpu());
  }

  atAttentionOut = impl::aten::toDevice(atAttentionOut).to(dtype);

  attentionOutAt.copy_(atAttentionOut);
  END_CALL_ACL_OP();
}

} // namespace OP_IMPL_NS
