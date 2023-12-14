/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presence_penalty,
                               diopiConstTensorHandle_t frequency_penalty, diopiConstTensorHandle_t p_token_ids, diopiConstTensorHandle_t p_token_counts,
                               diopiConstTensorHandle_t p_cumsum_seq_len, int p_max_len_in_batch) {
    AscendTensor asLogits(logits);                       // shape: [batch_size, p_max_len_in_batch]
    AscendTensor asPresencePenalty(presence_penalty);    // shape: [batch_size, ]
    AscendTensor asFrequencyPenalty(frequency_penalty);  // shape: [batch_size, ]
    AscendTensor asPTokenIds(p_token_ids);               // shape: [generated_tokens_num, ]
    AscendTensor asPTokenCounts(p_token_counts);         // shape: [generated_tokens_num, ]
    AscendTensor asPcumsumSeqLen(p_cumsum_seq_len);      // shape: [batch_size+1,]

    int64_t batch = asLogits.shape(0);
    for (int64_t i = 0; i < batch; ++i) {
        // int curBatchStartIndex = atPCumsumSeqLen[i].item<int>();
        // int curBatchEndIndex = atPCumsumSeqLen[i + 1].item<int>();
        void *PCumsumSeqLenCPU;  // 需要进行copy操作，将数据从GPU拷贝到CPU
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        CALL_ACLRT(aclrtMallocHost(&PCumsumSeqLenCPU, asPcumsumSeqLen.numel() * asPcumsumSeqLen.elemsize()));
        CALL_ACLRT(aclrtMemcpyAsync(PCumsumSeqLenCPU,
                                    asPcumsumSeqLen.numel() * asPcumsumSeqLen.elemsize(),
                                    asPcumsumSeqLen.data(),
                                    asPcumsumSeqLen.numel() * asPcumsumSeqLen.elemsize(),
                                    ACL_MEMCPY_DEVICE_TO_HOST,
                                    reinterpret_cast<aclrtStream>(stream)));
        CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));

        // 根据PCumsumSeqLenCPU获取每个batch的起始和终止索引
        int32_t curBatchStartIndex = *(reinterpret_cast<int32_t *>(PCumsumSeqLenCPU) + i);
        int32_t curBatchEndIndex = *(reinterpret_cast<int32_t *>(PCumsumSeqLenCPU) + i + 1);

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
        std::vector<int64_t> ithLogitsShape{asLogits.shape().begin() + 1, asLogits.shape().end()};
        AscendTensor ithLogits;
        makeTensor(ctx, ithLogits, ithLogitsShape, asLogits.dtype());

        diopiScalar_t scalarI;
        scalarI.stype = diopi_dtype_int32;
        scalarI.ival = i;
        AscendTensor tensorI;
        auto tempHandle = const_cast<diopiTensorHandle_t>(tensorI.tensorHandle());
        makeTensorFromScalar(ctx, &scalarI, &(tempHandle), diopi_dtype_int32, diopiDevice_t::diopi_device);
        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(ithLogits.tensorHandle()), logits, 0, tensorI.tensorHandle());

        // at::Tensor curLogits = atLogits[i].index_select(0, curTokenIds);
        std::vector<int64_t> curLogitsShape{
            curTokenIds.numel(),
        };  // 不确定curLogitsShape是否计算正确
        AscendTensor curLogits;
        makeTensor(ctx, curLogits, curLogitsShape, asLogits.dtype());
        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(curLogits.tensorHandle()), ithLogits.tensorHandle(), 0, curTokenIds.tensorHandle());

        // atFrequencyPenalty[i]
        std::vector<int64_t> tempShape{
            1,
        };  // 不确定tempShape是否计算正确
        AscendTensor ithFrequencyPenalty;
        makeTensor(ctx, ithFrequencyPenalty, tempShape, asFrequencyPenalty.dtype());
        diopiIndexSelect(
            ctx, const_cast<diopiTensorHandle_t>(ithFrequencyPenalty.tensorHandle()), asFrequencyPenalty.tensorHandle(), 0, tensorI.tensorHandle());

        // atPresencePenalty[i]
        AscendTensor ithPresencePenalty;
        makeTensor(ctx, ithPresencePenalty, tempShape, asPresencePenalty.dtype());
        diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(ithPresencePenalty.tensorHandle()), asPresencePenalty.tensorHandle(), 0, tensorI.tensorHandle());

        // curLogits = curLogits - curTokenCounts * atFrequencyPenalty[i] - atPresencePenalty[i];
        // 基于Frequency和TokenCounts对当前batch的Logits进行惩罚
        diopiMulInp(ctx, const_cast<diopiTensorHandle_t>(curTokenCounts.tensorHandle()), ithFrequencyPenalty.tensorHandle());
        diopiScalar_t alpha;
        alpha.stype = diopi_dtype_int32;
        alpha.ival = 1;
        diopiSubInp(ctx, const_cast<diopiTensorHandle_t>(curLogits.tensorHandle()), curTokenCounts.tensorHandle(), &alpha);
        diopiSubInp(ctx, const_cast<diopiTensorHandle_t>(curLogits.tensorHandle()), ithPresencePenalty.tensorHandle(), &alpha);

        // atLogits.index_put_({at::tensor(i), curTokenIds}, curLogits); #todo 需要index_put算子
        // 将惩罚后的当前批次的Logit重新写入
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
