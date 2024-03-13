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
#if 0
at::Tensor torchContextAttention(at::Tensor xq, at::Tensor xk, at::Tensor xv, int batchSize, int seqLen, int head, int dim) {
    c10::ScalarType dtype = xq.scalar_type();
    c10::Device device = xq.device();
    c10::Layout layout = xq.layout();
    xq = xq.view({batchSize, seqLen, head, dim}).transpose(1, 2);
    xk = xk.view({batchSize, seqLen, head, dim}).transpose(1, 2);
    xv = xv.view({batchSize, seqLen, head, dim}).transpose(1, 2);
    at::Tensor mask = op_api::tril(op_api::ones({seqLen, seqLen}, at::kFloat, layout, device)).unsqueeze(0).unsqueeze(0);
    mask.masked_fill_(mask == 0., -100000000.0);
    mask = mask.repeat({batchSize, head, 1, 1});
    at::Tensor scores = op_api::matmul(xq.to(at::kFloat), xk.transpose(2, 3).to(at::kFloat)) / std::sqrt(dim);
    at::Tensor output = op_api::matmul((scores + mask).softmax(-1), xv.to(at::kFloat)).transpose(1, 2).to(dtype);
    output = output.view({output.numel() / static_cast<int64_t>(head * dim), head, dim});
    return output;
}
#else
at::Tensor torchContextAttention(at::Tensor xq, at::Tensor xk, at::Tensor xv, int batchSize, int seqLen, int head, int dim) {
    c10::Device device = xq.device();
    c10::Layout layout = xq.layout();
    at::Tensor query = xq.view({batchSize, seqLen, static_cast<int64_t>(head * dim)});
    at::Tensor key = xk.view({batchSize, seqLen, static_cast<int64_t>(head * dim)});
    at::Tensor value = xv.view({batchSize, seqLen, static_cast<int64_t>(head * dim)});
    at::Tensor mask = op_api::tril(op_api::ones({seqLen, seqLen}, at::kBool, layout, device));
    mask = mask.repeat({batchSize, 1, 1});
    mask = op_api::logical_not(mask);
    at::Tensor output =
        op_api::npu_prompt_flash_attention(query, key, value, c10::nullopt, mask, c10::nullopt, head, 1 / std::sqrt(dim), 214748647, 0, "BSH", 0);
    return output.view({output.numel() / static_cast<int64_t>(head * dim), head, dim});
}
#endif
diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
    BEGIN_CALL_ACL_OP(out, q, k, v, bStartLoc, bSeqLen);
    int batch = bStartLocAt.size(0);
    int head = qAt.size(1);
    int dim = qAt.size(2);
    c10::Device device = qAt.device();
    c10::Layout layout = qAt.layout();
    at::Tensor bSeqLenCpu = bSeqLenAt.to(at::kCPU);
    at::Tensor bStartLocCpu = bStartLocAt.to(at::kCPU);
    for (int i = 0; i < batch; ++i) {
        int start = bStartLocCpu[i].item<int>();
        int end = start + bSeqLenCpu[i].item<int>();
        at::Tensor slice = op_api::arange(start, end, at::kLong, layout, device);
        at::Tensor values =
            torchContextAttention(at::index(qAt, {slice}), at::index(kAt, {slice}), at::index(vAt, {slice}), 1, bSeqLenCpu[i].item<int>(), head, dim);
        at::index_put_(outAt, {slice}, values);
    }
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
