#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>

#include <ATen/cuda/DeviceUtils.cuh>
# include "../../lmdeploy_kernel.h"


namespace ext {
namespace ops {

namespace {
    template<typename T>
    struct Vec_t {
        static constexpr int size = 0;
    };

    template<>
    struct Vec_t<float> {
        using Type                = float2;
        static constexpr int size = 2;
    };

    template<>
    struct Vec_t<half> {
        using Type                = uint32_t;
        static constexpr int size = 2;
    };

    inline __device__ float half_to_float(uint16_t h)
    {
        float f;
        asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
        return f;
    }

    inline __device__ uint32_t float2_to_half2(float2 f)
    {
        union {
            uint32_t u32;
            uint16_t u16[2];
        } tmp;
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
    #else
        asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
        asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
    #endif
        return tmp.u32;
    }

    inline __device__ float2 half2_to_float2(uint32_t v)
    {
        uint16_t lo, hi;
        asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
        return make_float2(half_to_float(lo), half_to_float(hi));
    }

    inline __device__ float2 rotary_embedding_coefficient(int zid, int rot_embed_dim, float base, float t_step)
    {
        const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
        return {cos(inv_freq), sin(inv_freq)};
    }

    inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
    {
        float2 rot_v;
        rot_v.x = coef.x * v.x - coef.y * v.y;
        rot_v.y = coef.x * v.y + coef.y * v.x;
        return rot_v;
    }

    inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
    {
        float2 fv     = half2_to_float2(v);
        float2 rot_fv = rotary_embedding_transform(fv, coef);
        return float2_to_half2(rot_fv);
    }

    inline __device__ void 
    apply_rotary_embedding(float2& q, float2& k, int tid, int rot_embed_dim, float base, float t_step)
    {
        if (2 * tid >= rot_embed_dim) {
            return;
        }
        const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, t_step);
        q               = rotary_embedding_transform(q, coef);
        k               = rotary_embedding_transform(k, coef);
    }

    inline __device__ void
    apply_rotary_embedding(uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, float base, float t_step)
    {
        if (2 * tid >= rot_embed_dim) {
            return;
        }
        const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, base, t_step);
        q               = rotary_embedding_transform(q, coef);
        k               = rotary_embedding_transform(k, coef);
    }

    template<typename T>
    __global__ void apply_gen_rotary_embedding(T* q, T* k, const float* step, int64_t local_head_num, int64_t local_kv_head_num, int64_t size_per_head,
                                float rotary_embedding_base, int64_t rotray_embedding_dim) {
        const int bs_id = blockIdx.x;
        const int head_id = blockIdx.y;
        const int tid     = threadIdx.x;
        using Vec_type                    = typename Vec_t<T>::Type;

        const int src_q_idx = bs_id * local_head_num * size_per_head + head_id * size_per_head + tid * 2;
        const int src_k_idx = bs_id * local_kv_head_num * size_per_head + head_id * size_per_head + tid * 2;

        Vec_type& q_buff = *reinterpret_cast<Vec_type*>(&q[src_q_idx]);
        Vec_type& k_buff = *reinterpret_cast<Vec_type*>(&k[src_k_idx]);
        float t_step = *(step + bs_id);
        apply_rotary_embedding(q_buff, k_buff, tid, rotray_embedding_dim, rotary_embedding_base, t_step);
    }

    template<typename T>
    __global__ void apply_prefill_rotary_embedding(T* q, T* k, int64_t step, int64_t local_head_num, int64_t local_kv_head_num, int64_t size_per_head,
                                float rotary_embedding_base, int64_t rotray_embedding_dim) {
        const int bs_id = blockIdx.x;
        const int head_id = blockIdx.y;
        const int tid     = threadIdx.x;
        using Vec_type                    = typename Vec_t<T>::Type;

        const int src_q_idx = bs_id * local_head_num * size_per_head + head_id * size_per_head + tid * 2;
        const int src_k_idx = bs_id * local_kv_head_num * size_per_head + head_id * size_per_head + tid * 2;

        Vec_type& q_buff = *reinterpret_cast<Vec_type*>(&q[src_q_idx]);
        Vec_type& k_buff = *reinterpret_cast<Vec_type*>(&k[src_k_idx]);
        float t_step = step + bs_id;
        apply_rotary_embedding(q_buff, k_buff, tid, rotray_embedding_dim, rotary_embedding_base, t_step);
    }

}

    void lmdeploy_gen_rotary_cuda(void* stream, void* q, void* k, const float* step, bool is_fp16,
                                int64_t local_head_num, int64_t local_kv_head_num, int64_t size_per_head, int64_t batch_size,
                                float rotary_embedding_base, int64_t rotray_embedding_dim) {
        assert(size_per_head % 64 == 0);
        dim3   block((size_per_head / 2 + 31) / 32 * 32);
        dim3   grid(batch_size, local_head_num);
        size_t smem_size = 0;
        if (is_fp16) {
            apply_gen_rotary_embedding<<<grid, block, smem_size, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<half*>(q),
                        reinterpret_cast<half*>(k), step, local_head_num, local_kv_head_num, size_per_head, rotary_embedding_base, rotray_embedding_dim);
        } else {
            apply_gen_rotary_embedding<<<grid, block, smem_size, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<float*>(q),
                        reinterpret_cast<float*>(k), step, local_head_num, local_kv_head_num, size_per_head, rotary_embedding_base, rotray_embedding_dim);
        }
    }

    void lmdeploy_prefill_rotary_cuda(void* stream, void* q, void* k, bool is_fp16, int64_t history_length, int64_t input_length,
                                int64_t local_head_num, int64_t local_kv_head_num, int64_t size_per_head,
                                float rotary_embedding_base, int64_t rotray_embedding_dim) {
        assert(size_per_head % 64 == 0);
        dim3   block((size_per_head / 2 + 31) / 32 * 32);
        dim3   grid(input_length, local_head_num);
        size_t smem_size = 0;
        if (is_fp16) {
            apply_prefill_rotary_embedding<<<grid, block, smem_size, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<half*>(q),
                        reinterpret_cast<half*>(k), history_length, local_head_num, local_kv_head_num, size_per_head, rotary_embedding_base, rotray_embedding_dim);
        } else {
            apply_prefill_rotary_embedding<<<grid, block, smem_size, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<float*>(q),
                        reinterpret_cast<float*>(k), history_length, local_head_num, local_kv_head_num, size_per_head, rotary_embedding_base, rotray_embedding_dim);
        }
    }

}  // namespace ops
}  // namespace ext