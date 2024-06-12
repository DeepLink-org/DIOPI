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

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    BEGIN_CALL_ACL_OP(attentionOut, q, k, bLoc, bStartLoc, bSeqLen);
    int batch = bLocAt.size(0);
    int head = qAt.size(1);
    int dim = qAt.size(2);
    qAt = qAt.reshape({batch, 1, head, dim}).transpose(1, 2);
    c10::ScalarType dtype = qAt.scalar_type();
    c10::Device device = qAt.device();
    c10::Layout layout = qAt.layout();
    for (int i = 0; i < batch; ++i) {
        int curSeqLen = bSeqLenAt[i].item<int>();
        int curSeqStartLoc = bStartLocAt[i].item<int>();
        at::Tensor kLoc = op_api::index_select(bLocAt[i], 0, op_api::arange(maxInputLen - curSeqLen, maxInputLen, at::kInt, layout, device));
        at::Tensor key = at::index(kAt, {kLoc}).view({1, curSeqLen, head, dim}).transpose(1, 2);
        at::Tensor outLoc = op_api::arange(curSeqStartLoc, curSeqStartLoc + curSeqLen, at::kInt, layout, device);
        at::Tensor values =
            (op_api::matmul(at::index(qAt, {torch::scalar_to_tensor(i)}).toType(at::kFloat), key.transpose(2, 3).toType(at::kFloat)) / std::sqrt(dim))
                .view({head, curSeqLen})
                .toType(dtype);
        at::index_put_(attentionOutAt, {at::Tensor(), outLoc}, values);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiPagedAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, diopiSize_t actualSeqLengths, int64_t numHeads, int64_t numKeyValueHeads, int64_t dim,
                                 diopiConstTensorHandle_t blockTable, int64_t blockSize) {
    BEGIN_CALL_ACL_OP(out, q, k, v, blockTable);
    at::IntArrayRef actSeqLen(actualSeqLengths.data, actualSeqLengths.len);
    TORCH_CHECK(actualSeqLengths.len == qAt.size(0), "The size of the first dimension of q must be equal to the length of actualSeqLengths!");
    TORCH_CHECK(actualSeqLengths.len == outAt.size(0), "The size of the first dimension of out must be equal to the length of actualSeqLengths!");
    if (qAt.dim() == 2) {
        qAt = impl::aten::viewStorage(qAt, {qAt.size(0), (int64_t)1, qAt.size(1)});
        outAt = impl::aten::viewStorage(outAt, {outAt.size(0), (int64_t)1, outAt.size(1)});
        kAt = impl::aten::viewStorage(kAt, {kAt.size(0), (int64_t)1, kAt.size(1)});
        vAt = impl::aten::viewStorage(vAt, {vAt.size(0), (int64_t)1, vAt.size(1)});
    }
    if (qAt.dim() == 3) {
        TORCH_CHECK(1 == qAt.size(1), "The size of the second dimension of q must be 1!");
        TORCH_CHECK(1 == outAt.size(1), "The size of the second dimension of out must be 1!");
    }
    double scaleValue = 1 / std::sqrt(dim);
    at::TensorList keyTensors = kAt;
    at::TensorList valueTensors = vAt;
    int64_t innerPrecise = 1;
    at::Tensor paddingMask, attenMask, dequantScale1, quantScale1, dequantScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, kvPaddingSize;
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnIncreFlashAttentionV4,
                                 qAt,
                                 keyTensors,
                                 valueTensors,
                                 paddingMask,
                                 attenMask,
                                 actSeqLen,
                                 dequantScale1,
                                 quantScale1,
                                 dequantScale2,
                                 quantScale2,
                                 quantOffset2,
                                 antiquantScale,
                                 antiquantOffset,
                                 blockTableAt,
                                 kvPaddingSize,
                                 numHeads,
                                 scaleValue,
                                 "BSH",
                                 numKeyValueHeads,
                                 blockSize,
                                 innerPrecise,
                                 outAt);
    END_CALL_ACL_OP()
}

}  // namespace OP_IMPL_NS
