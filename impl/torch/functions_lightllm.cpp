/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>
#include <torch/nn.h>
#include <torch/optim.h>

#include "context.h"
#include "helper.hpp"

extern "C" {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atDestLoc = impl::aten::buildATen(destLoc);
    atOut.index_put_({atDestLoc}, atK);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                               diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t pTokenIds, diopiConstTensorHandle_t pTokenCounts,
                               diopiConstTensorHandle_t pCumsumSeqLen, int pMaxLenInBatch) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atLogits = impl::aten::buildATen(logits);
    at::Tensor atPresencePenalty = impl::aten::buildATen(presencePenalty);
    at::Tensor atFrequencyPenalty = impl::aten::buildATen(frequencyPenalty);
    at::Tensor atPTokenIds = impl::aten::buildATen(pTokenIds);
    at::Tensor atPTokenCounts = impl::aten::buildATen(pTokenCounts);
    at::Tensor atPCumsumSeqLen = impl::aten::buildATen(pCumsumSeqLen);

    int batch = atLogits.size(0);
    for (int i = 0; i < batch; ++i) {
        int curBatchStartIndex = atPCumsumSeqLen[i].item<int>();
        int curBatchEndIndex = atPCumsumSeqLen[i + 1].item<int>();
        at::Tensor curTokenIds = atPTokenIds.slice(0, curBatchStartIndex, curBatchEndIndex);
        at::Tensor curTokenCounts = atPTokenCounts.slice(0, curBatchStartIndex, curBatchEndIndex);
        at::Tensor curLogits = atLogits[i].index_select(0, curTokenIds);
        curLogits = curLogits - curTokenCounts * atFrequencyPenalty[i] - atPresencePenalty[i];
        atLogits.index_put_({at::tensor(i), curTokenIds}, curLogits);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
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

    atQ = atQ.reshape({batch, 1, head, dim}).transpose(1, 2);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < atBSeqLen[i].item<int>(); ++j) {
            at::Tensor kLoc = atBLoc[i][maxInputLen - atBSeqLen[i].item<int>() + j];
            int outLoc = atBStartLoc[i].item<int>() + j;
            at::Tensor key = atK.index({kLoc}).reshape({1, 1, head, dim}).transpose(1, 2);
            at::Tensor values = (at::matmul(atQ.index({i}), key.transpose(2, 3)) / std::sqrt(dim)).squeeze().reshape(head);
            atAttentionOut.index_put_({torch::indexing::Slice(), outLoc}, values);
        }
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                               int maxInputLen, int otherKVIndex) {
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

    // softmax
    at::Tensor prob = at::empty_like(atLogics);
    for (int i = 0; i < batch; ++i) {
        int start = atBStartLoc[i].item<int>();
        int end = start + atBSeqLen[i].item<int>();
        prob.slice(1, start, end) = atLogics.slice(1, start, end).reshape({head, -1}).softmax(-1);
    }

    // reduce_V
    for (int i = 0; i < batch; ++i) {
        std::vector<at::Tensor> vOut;
        for (int j = 0; j < atBSeqLen[i].item<int>(); ++j) {
            int vLoc = atBLoc[i][maxInputLen - atBSeqLen[i].item<int>() + j].item<int>();
            vOut.emplace_back(atV[vLoc]);
        }

        at::Tensor V = at::cat(vOut, 0).view({1, atBSeqLen[i].item<int>(), head, dim}).transpose(1, 2);
        at::Tensor P = prob.slice(1, atBStartLoc[i].item<int>(), atBStartLoc[i].item<int>() + atBSeqLen[i].item<int>())
                           .reshape({head, 1, 1, atBSeqLen[i].item<int>()})
                           .transpose(0, 1);
        atOut[i] = at::matmul(P, V).view({head, dim});
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

at::Tensor torchContextAttention(at::Tensor xq, at::Tensor xk, at::Tensor xv, int batchSize, int seqLen, int head, int dim) {
    xq = xq.view({batchSize, seqLen, head, dim});
    xk = xk.view({batchSize, seqLen, head, dim});
    xv = xv.view({batchSize, seqLen, head, dim});
    at::Tensor mask = at::tril(at::ones({seqLen, seqLen})).unsqueeze(0).unsqueeze(0).to(at::kCUDA);
    mask.masked_fill_(mask == 0., -100000000.0);
    mask = mask.repeat({batchSize, head, 1, 1});
    at::Tensor keys = xk;
    at::Tensor values = xv;
    xq = xq.transpose(1, 2);
    keys = keys.transpose(1, 2);
    values = values.transpose(1, 2);
    at::Tensor scores = at::matmul(xq, keys.transpose(2, 3)) / std::sqrt(dim);
    scores = at::softmax((scores.to(at::kFloat) + mask), -1).to(xq.scalar_type());
    at::Tensor output = at::matmul(scores, values).transpose(1, 2).contiguous().view({-1, head, dim});
    return output;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atQ = impl::aten::buildATen(q);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atV = impl::aten::buildATen(v);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);

    int batch = atBStartLoc.size(0);
    int head = atQ.size(1);
    int dim = atQ.size(2);
    for (int i = 0; i < batch; ++i) {
        int start = atBStartLoc[i].item<int>();
        int end = start + atBSeqLen[i].item<int>();
        atOut.slice(0, start, end) =
            torchContextAttention(atQ.slice(0, start, end), atK.slice(0, start, end), atV.slice(0, start, end), 1, atBSeqLen[i].item<int>(), head, dim);
    }
    return diopiSuccess;
}

}  // extern "C"
