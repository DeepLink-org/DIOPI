/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include <c10/core/TensorImpl.h>
// #include <torch/csrc/autograd/generated/variable_type.h>
// #include <torch/csrc/autograd/generated/variable_type.h>
#include <acl/acl.h>
#include <torch/torch.h>

namespace OP_IMPL_NS {
// at::Tensor toDevice(at::Tensor tensor) {
//     int devId_ = 0;
//     ::aclrtGetDevice(&devId_);
//     auto options = at::TensorOptions(c10::Device(at::DeviceType::XLA,
//     devId_)).dtype(tensor.dtype()); return
//     fromPreAllocated(tensor.data_ptr(), tensor.dim(), tensor.strides(),
//     options);
// }

diopiError_t diopiTokenAttentionInference(
    diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut,
    diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
    diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc,
    diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
  impl::aten::setCurCtx(ctx);
  at::Tensor atQ = impl::aten::buildATen(q);
  at::Tensor atK = impl::aten::buildATen(k);
  at::Tensor atBLoc = impl::aten::buildATen(bLoc);
  at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
  at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);
  at::Tensor atAttentionOut = impl::aten::buildATen(attentionOut);

  int batch = atBLoc.size(0);
  int head = atQ.size(1);
  int dim = atQ.size(2);
  std::cout << "batch=" << batch << ", head=" << head << ", dim=" << dim
            << std::endl;
  std::cout << "atBSeqLen=" << atBSeqLen << std::endl;

  atQ = atQ.reshape({batch, 1, head, dim}).transpose(1, 2);
  // std::cout << "atQ=" << atQ << std::endl;
  for (int i = 0; i < batch; ++i) {
    std::cout << "i=" << i << std::endl;
    int curSeqLen = atBSeqLen.cpu()[i].item<int>();
    int curSeqStartLoc = atBStartLoc.cpu()[i].item<int>();
    at::Tensor kLoc = atBLoc.cpu()[i].index_select(
        0, at::arange(maxInputLen - curSeqLen, maxInputLen));
    // std::cout << "kLoc0=" << kLoc0 << std::endl;
    // std::cout << "atQ.device()=" << atQ.device() << std::endl;
    // at::Tensor kLoc = kLoc0.to(atQ.device());
    std::cout << "kLoc="
              << "kLoc value is too more." << std::endl;
    // at::Tensor key = atK.index({kLoc}).view({1, curSeqLen, head,
    // dim}).transpose(1, 2);
    torch::List<c10::optional<at::Tensor>> indicesAtList;
    auto kLoc1 = kLoc.to(at::ScalarType::Long);
    auto kLoc2 = kLoc1.to(atBSeqLen.device());
    auto kLoc3 = aten::toDevice(kLoc1);
    std::cout << "atBSeqLen.device()=" << atBSeqLen.device()
              << ", kLoc2.device()=" << kLoc2.device()
              << ", kLoc1.device()=" << kLoc1.device()
              << ", kLoc3.device()=" << kLoc3.device() << std::endl;
    // auto kLoc3 = kLoc1.to(at::Device(at::DeviceType::XLA, 0));
    // auto kLoc2 = kLoc1.to();
    indicesAtList.push_back(kLoc1);
    for (auto item : indicesAtList) {
      std::cout << "item=";
      std::cout << item.get() << std::endl;
    }
    std::cout << "atK.device()=" << atK.device()
              << ", kLoc1.device()=" << kLoc1.device() << std::endl;
    auto cpu = atK.cpu();
    // std::cout << "======atK=" << atK << std::endl;
    // std::cout << "======kLoc3=" << kLoc3 << std::endl;
    // at::Tensor key = acl_op::index(cpu, indicesAtList);
    at::Tensor key = at::index(cpu, indicesAtList);
    std::cout << "======key=" << "key is too more." << std::endl;
    at::Tensor key1 = key.view({1, curSeqLen, head, dim}).transpose(1, 2);
    at::Tensor outLoc = at::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen);
    std::cout << "outLoc=" << outLoc << std::endl;
    std::cout << "Before transpose, key shape: " << key.sizes() << std::endl;

    auto a = atQ.cpu().index({i});
    std::cout << "a.device()=" << a.device() << std::endl;
    a = aten::toDevice(a);
    std::cout << "========a.device()=" << a.device() << std::endl;
    auto b = aten::toDevice(key1.transpose(2, 3));
    std::cout << "========b.device()=" << b.device() << std::endl;
    at::Tensor values =
        (at::matmul(a, b) / std::sqrt(dim))
            .view({head, curSeqLen});
    std::cout << "values=" << values << std::endl;
    atAttentionOut.index_put_({torch::indexing::Slice(), outLoc.cpu()}, values);
    std::cout << "finish i=" << i << std::endl;
  }
  impl::aten::unsetCurCtx();
  return diopiSuccess;
}

} // namespace OP_IMPL_NS
