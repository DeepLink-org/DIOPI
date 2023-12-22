/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presencePenalty,
                               diopiConstTensorHandle_t frequencyPenalty, diopiConstTensorHandle_t pTokenIds, diopiConstTensorHandle_t pTokenCounts,
                               diopiConstTensorHandle_t pCumsumSeqLen, int pMaxLenInBatch) {
    AscendTensor asLogits(logits);                      // shape: [batch_size, pMaxLenInBatch]
    AscendTensor asPresencePenalty(presencePenalty);    // shape: [batch_size, ]
    AscendTensor asFrequencyPenalty(frequencyPenalty);  // shape: [batch_size, ]
    AscendTensor asPTokenIds(pTokenIds);                // shape: [generated_tokens_num, ]
    AscendTensor asPTokenCounts(pTokenCounts);          // shape: [generated_tokens_num, ]
    AscendTensor asPcumsumSeqLen(pCumsumSeqLen);        // shape: [batch_size+1,]

    void *pCumsumSeqLenCpu = nullptr;  // 需要进行copy操作，将数据从GPU拷贝到CPU
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    CALL_ACLRT(aclrtMallocHost(&pCumsumSeqLenCpu, asPcumsumSeqLen.numel() * asPcumsumSeqLen.elemsize()));
    CALL_ACLRT(aclrtMemcpyAsync(pCumsumSeqLenCpu,
                                asPcumsumSeqLen.numel() * asPcumsumSeqLen.elemsize(),
                                asPcumsumSeqLen.data(),
                                asPcumsumSeqLen.numel() * asPcumsumSeqLen.elemsize(),
                                ACL_MEMCPY_DEVICE_TO_HOST,
                                reinterpret_cast<aclrtStream>(stream)));
    CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));

    int64_t batch = asLogits.shape(0);
    for (int64_t i = 0; i < batch; ++i) {
        // int curBatchStartIndex = atPCumsumSeqLen[i].item<int>();
        // int curBatchEndIndex = atPCumsumSeqLen[i + 1].item<int>();
        // 根据pCumsumSeqLenCpu获取每个batch的起始和终止索引
        int32_t curBatchStartIndex = *(reinterpret_cast<int32_t *>(pCumsumSeqLenCpu) + i);
        int32_t curBatchEndIndex = *(reinterpret_cast<int32_t *>(pCumsumSeqLenCpu) + i + 1);

        // at::Tensor curTokenIds = atPTokenIds.slice(0, curBatchStartIndex, curBatchEndIndex);
        // 根据起始和终止索引，在PTokenIds中提取当前batch的TokenId序列
        AscendTensor curTokenIds;
        std::vector<int64_t> curTokenIdsShape{
            curBatchEndIndex - curBatchStartIndex,
        };
        makeTensor(ctx, curTokenIds, curTokenIdsShape, asPTokenIds.dtype());
        diopiSlice(ctx, const_cast<diopiTensorHandle_t>(curTokenIds.tensorHandle()), asPTokenIds.tensorHandle(), 0, curBatchStartIndex, curBatchEndIndex, 1);

        // at::Tensor curTokenCounts = atPTokenCounts.slice(0, curBatchStartIndex, curBatchEndIndex);
        // 根据起始和终止索引，在PTokenCounts中提取当前batch的对应计数
        AscendTensor curTokenCounts;
        makeTensor(ctx, curTokenCounts, curTokenIdsShape, asPTokenCounts.dtype());
        diopiSlice(
            ctx, const_cast<diopiTensorHandle_t>(curTokenCounts.tensorHandle()), asPTokenCounts.tensorHandle(), 0, curBatchStartIndex, curBatchEndIndex, 1);

        //  atLogits[i]
        // std::vector<int64_t> ithLogitsShape{asLogits.shape().begin() + 1, asLogits.shape().end()};
        std::vector<int64_t> ithLogitsShape{asLogits.shape(1)};
        AscendTensor ithLogits;
        makeTensor(ctx, ithLogits, ithLogitsShape, asLogits.dtype());
        diopiScalar_t scalarI;
        scalarI.stype = diopi_dtype_int32;
        scalarI.ival = i;

        diopiTensorHandle_t tensorIHandle;
        makeTensorFromScalar(ctx, &scalarI, &tensorIHandle, diopi_dtype_int32, diopiDevice_t::diopi_device);
        AscendTensor tensorI(tensorIHandle);

        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(ithLogits.tensorHandle()), logits, 0, tensorI.tensorHandle());

        // at::Tensor curLogits = atLogits[i].index_select(0, curTokenIds);
        std::vector<int64_t> curLogitsShape{
            curTokenIds.numel(),
        };

        AscendTensor curLogits;
        makeTensor(ctx, curLogits, curLogitsShape, asLogits.dtype());
        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(curLogits.tensorHandle()), ithLogits.tensorHandle(), 0, curTokenIds.tensorHandle());

        // atFrequencyPenalty[i]
        std::vector<int64_t> tempShape{
            1,
        };
        AscendTensor ithFrequencyPenalty;
        makeTensor(ctx, ithFrequencyPenalty, tempShape, asFrequencyPenalty.dtype());
        diopiIndexSelect(
            ctx, const_cast<diopiTensorHandle_t>(ithFrequencyPenalty.tensorHandle()), asFrequencyPenalty.tensorHandle(), 0, tensorI.tensorHandle());

        // atPresencePenalty[i]clear

        AscendTensor ithPresencePenalty;
        makeTensor(ctx, ithPresencePenalty, tempShape, asPresencePenalty.dtype());
        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(ithPresencePenalty.tensorHandle()), asPresencePenalty.tensorHandle(), 0, tensorI.tensorHandle());

        // curLogits = curLogits - curTokenCounts * atFrequencyPenalty[i] - atPresencePenalty[i];
        // 基于Frequency和TokenCounts对当前batch的Logits进行惩罚
        AscendTensor asFrequencyPenalty;
        makeTensor(ctx, asFrequencyPenalty, {curTokenCounts.shape()}, asLogits.dtype());
        diopiMul(ctx,
                 const_cast<diopiTensorHandle_t>(asFrequencyPenalty.tensorHandle()),
                 const_cast<diopiTensorHandle_t>(curTokenCounts.tensorHandle()),
                 ithFrequencyPenalty.tensorHandle());
        diopiScalar_t alpha;
        alpha.stype = diopi_dtype_int32;
        alpha.ival = 1;
        diopiSubInp(ctx, const_cast<diopiTensorHandle_t>(curLogits.tensorHandle()), asFrequencyPenalty.tensorHandle(), &alpha);
        diopiSubInp(ctx, const_cast<diopiTensorHandle_t>(curLogits.tensorHandle()), ithPresencePenalty.tensorHandle(), &alpha);
        // atLogits.index_put_({at::tensor(i), curTokenIds}, curLogits); #todo 需要index_put算子实现后进行重构
        // 将惩罚后的当前批次的Logit重新写入全局的Logits
        AscendTensor indexedNegLogits;
        makeTensor(ctx, indexedNegLogits, curTokenIds.shape(), asLogits.dtype());
        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(indexedNegLogits.tensorHandle()), ithLogits.tensorHandle(), 0, curTokenIds.tensorHandle());
        AclOpRunner<1, 1>("Neg", ctx).addInput(indexedNegLogits).addOutput(indexedNegLogits).run();
        AclOpRunner<3, 1>("InplaceIndexAdd", ctx)
            .addInput(ithLogits)
            .addInput(curTokenIds)
            .addInput(indexedNegLogits)
            .setAttr("axis", 0)
            .addOutput(ithLogits)
            .run();
        AclOpRunner<3, 1>("InplaceIndexAdd", ctx).addInput(ithLogits).addInput(curTokenIds).addInput(curLogits).setAttr("axis", 0).addOutput(ithLogits).run();
        AclOpRunner<3, 1>("InplaceUpdate", ctx).addInput(asLogits).addConstInput({i}).addInput(ithLogits).addOutput(asLogits).run();
    }
    CALL_ACLRT(aclrtFreeHost(pCumsumSeqLenCpu));
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
