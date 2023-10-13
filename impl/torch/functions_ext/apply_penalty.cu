#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/cuda/Loops.cuh>

#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"

#include "../cuda_helpers.h"
#include "ext_common.h"


using namespace cuda::helper;
namespace ext {
namespace ops {

    using namespace at;

    __global__ void _fwd_kernel_apply_penalty(
        at::Tensor Logits, // Logits是模型的输出，它是一个形状为[batch_size, sequence_length, vocab_size]的张量
        at::Tensor presence_penalty, // presence_penalty的形状是(batch_size)，每个输入序列都有一个单独的存在性惩罚值和频率惩罚值
        at::Tensor frequency_penalty, // frequency_penalty与presence_penalty类似，形状是(batch_size)
        at::Tensor p_token_ids, // p_token_ids: (total_tokens_in_batch,)
        at::Tensor p_token_counts, // p_token_counts: (total_tokens_in_batch,)
        at::Tensor p_cumsum_seq_len, // previous cumulative sum of sequence lengths"，即在批处理中每个序列的累积长度，其形状为 (batch_size,)，
                                    // 通常用于在处理变长序列中，计算序列中每个元素在扁平化张量中的位置
        int64_t stride_logit_b,
        int64_t stride_logit_s,
        int BLOCK_P // BLOCK_P是一个常量，表示每个线程块的大小，即每个内核处理的数据量
    ) {
        // 获取当前序列的ID，由于使用一维网格，每个线程块负责处理一个序列，blockIdx.x和序列ID(cur_seq)在这个例子中是等价的
        int cur_seq = blockIdx.x;

        // 使用索引操作，获取当前序列的频率惩罚和存在惩罚
        float cur_freqency = frequency_penalty[cur_seq].item<float>();
        float cur_presence = presence_penalty[cur_seq].item<float>();

        // 获取当前序列在扁平化序列中的的起始索引和结束索引，注意，每个序列长度不是等长的
        int cur_seq_start_index = p_cumsum_seq_len[cur_seq].item<int>();
        int cur_seq_end_index = p_cumsum_seq_len[cur_seq + 1].item<int>();

        // 计算当前序列的令牌ID和令牌计数
        int cur_seq_id_offset = cur_seq_start_index + threadIdx.x; // 创建一个偏移量的张量，用于后续计算中对序列进行索引操作
        int batch_ids = (cur_seq_id_offset < cur_seq_end_index) ? p_token_ids[cur_seq_id_offset].item<int>() : 0;
        int batch_ids_count = (cur_seq_id_offset < cur_seq_end_index) ? p_token_counts[cur_seq_id_offset].item<int>() : 0;

        // 在Logits中找到对应的位置，并加载相应的对数概率
        float* row_start_ptr = Logits[cur_seq].data_ptr<float>();
        float* cur_offset = row_start_ptr + batch_ids * stride_logit_s;
        float cur_logits = (cur_seq_id_offset < cur_seq_end_index) ? *cur_offset : 0.0f;

        // 计算预惩罚的对数概率
        // cur_logits是float类型，表示当前词汇的logits量(即未归一化的概率分布)
        // batch_ids_count是float类型，表示当前词汇在批次中出现的次数，这个值用于计算频率惩罚
        // cur_freqency是float类型，表示当前批次的评论惩罚值
        float freq_logits = cur_logits - batch_ids_count * cur_freqency;
        float pre_logits = freq_logits - cur_presence;

        // 将结果存储回Logits中的相应位置
        if (cur_seq_id_offset < cur_seq_end_index) {
            float* output_ptr = Logits[cur_seq].data_ptr<float>() + batch_ids * stride_logit_s;
            *output_ptr = pre_logits;
        }
    }
}  // namespace ops
}  // namespace ext