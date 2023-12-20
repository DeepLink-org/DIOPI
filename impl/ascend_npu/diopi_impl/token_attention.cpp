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

void printTensor(at::Tensor& b) {
      std::cout << "b.device()=" << b.device() << std::endl;
    std::cout << "b.sizes()=" << b.sizes() << std::endl;
    std::cout << "b.dtype()=" << b.dtype() << std::endl;
}

diopiError_t diopiTokenAttentionInference(
    diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut,
    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
    diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc,
    diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
  // impl::aten::setCurCtx(ctx);
  BEGIN_CALL_ACL_OP(attentionOut);
  at::Tensor atQ = impl::aten::buildATen(q);
  at::Tensor atK = impl::aten::buildATen(k);
  at::Tensor atBLoc = impl::aten::buildATen(bLoc);
  at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
  at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);
  at::Tensor atAttentionOut = impl::aten::buildATen(attentionOut);

  auto x = atAttentionOut.cpu();

  int batch = atBLoc.size(0);
  int head = atQ.size(1);
  int dim = atQ.size(2);

  atQ = atQ.reshape({batch, 1, head, dim}).transpose(1, 2);
  for (int i = 0; i < batch; ++i) {
    std::cout << "i=" << i << std::endl;
    int curSeqLen = atBSeqLen.cpu()[i].item<int>();
    int curSeqStartLoc = atBStartLoc.cpu()[i].item<int>();
    at::Tensor kLoc = atBLoc.cpu()[i].index_select(
        0, at::arange(maxInputLen - curSeqLen, maxInputLen));
    auto kLoc1 = kLoc.to(at::ScalarType::Long);
    auto cpu = atK.cpu();
    at::Tensor key = at::index(cpu, {kLoc1});
    at::Tensor key1 = key.view({1, curSeqLen, head, dim}).transpose(1, 2);
    at::Tensor outLoc = at::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen);

    auto a = atQ.cpu().index({i}).to(at::ScalarType::Float);
    if (a.dtype() != x.dtype()) {
      x = x.to(a.dtype());
    }
    auto b = key1.transpose(2, 3).cpu().to(at::ScalarType::Float);
    at::Tensor values =
        (at::matmul(a, b) / std::sqrt(dim)).view({head, curSeqLen});
    x.index_put_({torch::indexing::Slice(), outLoc.cpu()}, values);
    std::cout << "finish i=" << i << std::endl;
  }
  std::cout << "out for loop" << std::endl;

  atAttentionOut = aten::toDevice(x).to(atQ.dtype());

  std::cout << "finish all." << std::endl;
  END_CALL_ACL_OP();
}

} // namespace OP_IMPL_NS
