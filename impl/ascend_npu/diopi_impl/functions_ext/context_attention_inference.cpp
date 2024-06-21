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
    at::Tensor scores = op_api::matmul(xq.to(at::kFloat, true), xk.transpose(2, 3).to(at::kFloat, true)) / std::sqrt(dim);
    at::Tensor output = op_api::matmul((scores + mask).softmax(-1), xv.to(at::kFloat, true)).transpose(1, 2).to(dtype, true);
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
        at::Tensor slice = op_api::arange(start, end, at::kLong, layout, device);
        at::Tensor values =
            torchContextAttention(at::index(qAt, {slice}), at::index(kAt, {slice}), at::index(vAt, {slice}), 1, bSeqLenAt[i].item<int>(), head, dim);
        at::index_put_(outAt, {slice}, values);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiPromptFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t query, diopiConstTensorHandle_t key,
                                       diopiConstTensorHandle_t value, diopiConstTensorHandle_t attenMask, diopiSize_t actualSeqLengths, int64_t maxInputLen,
                                       int64_t numHeads, int64_t numKeyValueHeads, int64_t dim) {
    BEGIN_CALL_ACL_OP(out, query, key, value, attenMask);
    at::IntArrayRef actSeqLen(actualSeqLengths.data, actualSeqLengths.len);
    if (queryAt.dim() == 2) {
        queryAt = impl::aten::viewStorage(queryAt, {actualSeqLengths.len, maxInputLen, queryAt.size(1)});
        outAt = impl::aten::viewStorage(outAt, {actualSeqLengths.len, maxInputLen, outAt.size(1)});
        keyAt = impl::aten::viewStorage(keyAt, {actualSeqLengths.len, maxInputLen, keyAt.size(1)});
        valueAt = impl::aten::viewStorage(valueAt, {actualSeqLengths.len, maxInputLen, valueAt.size(1)});
    }
    double scaleValue = 1 / std::sqrt(dim);
    int64_t preTokens = 2147473647;
    int64_t nextTokens = 0;
    at::Tensor paddingMask;
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnPromptFlashAttention,
                                 queryAt,
                                 keyAt,
                                 valueAt,
                                 paddingMask,
                                 attenMaskAt,
                                 actSeqLen,
                                 numHeads,
                                 scaleValue,
                                 preTokens,
                                 nextTokens,
                                 "BSH",
                                 numKeyValueHeads,
                                 outAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
