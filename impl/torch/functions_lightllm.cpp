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
    auto atOut = impl::aten::buildATen(out);
    auto atValues = impl::aten::buildATen(k);
    auto atIndex = impl::aten::buildATen(destLoc);
    torch::List<c10::optional<at::Tensor>> atIndicesList;
    atIndicesList.emplace_back(atIndex);

    impl::aten::invokeATenFuncInp(ctx, at::index_put_, atOut, atIndicesList, atValues, false);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                               diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t pTokenIds, diopiConstTensorHandle_t pTokenCounts,
                               diopiConstTensorHandle_t pCumsumSeqLen, int pMaxLenInBatch) {
    impl::aten::setCurCtx(ctx);
    auto atLogits = impl::aten::buildATen(logits);
    auto atPresencePenalty = impl::aten::buildATen(presencePenalty);
    auto atFrequencyPenalty = impl::aten::buildATen(frequencyPenalty);
    auto atPTokenIds = impl::aten::buildATen(pTokenIds);
    auto atPTokenCounts = impl::aten::buildATen(pTokenCounts);
    auto atPCumsumSeqLen = impl::aten::buildATen(pCumsumSeqLen);

    int batch = atLogits.size(0);
    for (int i = 0; i < batch; ++i) {
        int curBatchStartIndex = atPCumsumSeqLen[i].item<int>();
        int curBatchEndIndex = atPCumsumSeqLen[i + 1].item<int>();
        auto curTokenIds = at::slice(atPTokenIds, 0, curBatchStartIndex, curBatchEndIndex, 1);
        auto curTokenCounts = at::slice(atPTokenCounts, 0, curBatchStartIndex, curBatchEndIndex, 1);
        auto curLogits = at::index_select(atLogits[i], 0, curTokenIds);
        curLogits = curLogits - curTokenCounts * atFrequencyPenalty[i] - atPresencePenalty[i];

        torch::List<c10::optional<at::Tensor>> atIndicesList;
        atIndicesList.emplace_back(at::tensor(i));
        atIndicesList.emplace_back(curTokenIds);
        impl::aten::invokeATenFuncInp(ctx, at::index_put_, atLogits, atIndicesList, curLogits, false);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    // def token_attention1(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    // batch, head, dim = B_Loc.shape[0], q.shape[1], q.shape[2]
    // xq = q.view(batch, 1, head, dim).transpose(1, 2)
    // for i in range(batch):
    //     for j in range(B_Seqlen[i]):
    //         k_loc = B_Loc[i][max_input_len-B_Seqlen[i]+j]
    //         out_loc = B_Start_Loc[i] + j
    //         key = k[k_loc, :].view(1, 1, head, dim).transpose(1, 2)
    //         att_out[:, out_loc] = (torch.matmul(xq[i, :], key.transpose(2, 3)) / math.sqrt(dim)).squeeze().reshape(head)
    // return

    impl::aten::setCurCtx(ctx);
    auto atQ = impl::aten::buildATen(q);
    auto atK = impl::aten::buildATen(k);
    auto atBLoc = impl::aten::buildATen(bLoc);
    auto atBStartLoc = impl::aten::buildATen(bStartLoc);
    auto atBSeqLen = impl::aten::buildATen(bSeqLen);
    auto atAttentionOut = impl::aten::buildATen(attentionOut);

    int batch = atBLoc.size(0);
    int head = atQ.size(1);
    int dim = atQ.size(2);

    std::vector<int64_t> atQShape = {batch, 1, head, dim};
    atQ = atQ.reshape(atQShape).transpose(1, 2);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < atBSeqLen[i].item<int>(); ++j) {
            auto kLoc = atBLoc[i][maxInputLen - atBSeqLen[i].item<int>() + j];
            auto outLoc = atBStartLoc[i].item<int>() + j;
            auto key = atK.index({kLoc}).reshape({1, 1, head, dim}).transpose(1, 2);
            // auto values = (at::matmul(atQ.index({i}), key.transpose(2, 3)) / std::sqrt(dim)).squeeze().reshape(head);
            auto values = (at::matmul(atQ.index({i}), key.transpose(2, 3)) / std::sqrt(dim)).squeeze();
            std::cout << "values.shape: " << values.sizes() << std::endl;
            auto test = atAttentionOut.index({torch::indexing::Slice(), outLoc});
            std::cout << "test.shape: " << test.sizes() << std::endl;
            // atAttentionOut.index_put_({torch::indexing::Slice(), outLoc}, values);
            // atAttentionOut.index_put_({torch::indexing::Slice(), outLoc}, values);
        }
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqlen,
                                               int maxInputLen, int otherKVIndex) {
    return diopiSuccess;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
    return diopiSuccess;
}

}  // extern "C"
