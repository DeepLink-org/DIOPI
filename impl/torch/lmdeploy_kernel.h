/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#ifndef IMPL_TORCH_LMDEPLOY_KERNEL_H_
#define IMPL_TORCH_LMDEPLOY_KERNEL_H_

namespace ext {
namespace ops {

void lmdeploy_gen_rotary_cuda(void* stream, void* q, void* k, const float* step, bool is_fp16, int64_t local_head_num, int64_t local_kv_head_num,
                              int64_t size_per_head, int64_t batch_size, float rotary_embedding_base, int64_t rotray_embedding_dim);

void lmdeploy_prefill_rotary_cuda(void* stream, void* q, void* k, bool is_fp16, int64_t history_length, int64_t input_length, int64_t local_head_num,
                                  int64_t local_kv_head_num, int64_t size_per_head, float rotary_embedding_base, int64_t rotray_embedding_dim);
}  // namespace ops
}  // namespace ext

#endif  // IMPL_TORCH_EXT_KERNEL_H_
