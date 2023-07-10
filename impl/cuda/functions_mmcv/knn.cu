/**
 * @file knn.cu
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cuda_runtime.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include "../cuda_helper.hpp"
#include "../helper.hpp"

namespace impl {

namespace cuda {
inline __device__ void swap_float(float *x, float *y) {
    float tmp = *x;
    *x = *y;
    *y = tmp;
}

inline __device__ void swap_int(int *x, int *y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ void reheap(float *dist, int *idx, int k) {
    int root = 0;
    int child = root * 2 + 1;
    while (child < k) {
        if (child + 1 < k && dist[child + 1] > dist[child]) child++;
        if (dist[root] > dist[child]) return;
        swap_float(&dist[root], &dist[child]);
        swap_int(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}

__device__ void heap_sort(float *dist, int *idx, int k) {
    int i;
    for (i = k - 1; i > 0; i--) {
        swap_float(&dist[0], &dist[i]);
        swap_int(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}

// input: xyz (b, n, 3) new_xyz (b, m, 3)
// output: idx (b, m, nsample) dist2 (b, m, nsample)
template <typename T>
__global__ void knn_forward_cuda_kernel(int b, int n, int m, int nsample, const void *xyz_, const void *new_xyz_, int *__restrict__ idx, void *dist2_) {
    int bs_idx = blockIdx.y;
    const T *xyz = static_cast<const T *>(xyz_);
    const T *new_xyz = static_cast<const T *>(new_xyz_);
    T *dist2 = static_cast<T *>(dist2_);
    CUDA_1D_KERNEL_LOOP(pt_idx, m) {
        if (bs_idx >= b) return;

        new_xyz += bs_idx * m * 3 + pt_idx * 3;
        xyz += bs_idx * n * 3;
        idx += bs_idx * m * nsample + pt_idx * nsample;
        dist2 += bs_idx * m * nsample + pt_idx * nsample;

        T new_x = new_xyz[0];
        T new_y = new_xyz[1];
        T new_z = new_xyz[2];

        float best_dist[100];
        int best_idx[100];
        for (int i = 0; i < nsample; i++) {
            best_dist[i] = 1e10;
            best_idx[i] = 0;
        }
        for (int i = 0; i < n; i++) {
            T x = xyz[i * 3 + 0];
            T y = xyz[i * 3 + 1];
            T z = xyz[i * 3 + 2];
            T d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            if (d2 < best_dist[0]) {
                best_dist[0] = d2;
                best_idx[0] = i;
                reheap(best_dist, best_idx, nsample);
            }
        }
        heap_sort(best_dist, best_idx, nsample);
        for (int i = 0; i < nsample; i++) {
            idx[i] = best_idx[i];
            dist2[i] = best_dist[i];
        }
    }
}
}  // namespace cuda

}  // namespace impl

diopiError_t diopiKnn(diopiContextHandle_t ctx, diopiTensorHandle_t xyz_, diopiTensorHandle_t new_xyz_, diopiTensorHandle_t idx_, diopiTensorHandle_t dist2_,
                      int64_t b, int64_t n, int64_t m, int64_t nsample) {
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param idx: (B, m, nsample)

    auto xyz = impl::cuda::makeTensor(xyz_);
    auto new_xyz = impl::cuda::makeTensor(new_xyz_);
    auto idx = impl::cuda::makeTensor(idx_);
    auto dist2 = impl::cuda::makeTensor(dist2_);

    // at::cuda::CUDAGuard device_guard(new_xyz.device());
    auto stream = impl::cuda::getStream(ctx);

    // blockIdx.x(col), blockIdx.y(row)
    dim3 blocks(GET_BLOCKS(m, THREADS_PER_BLOCK), b);
    dim3 threads(THREADS_PER_BLOCK);

    DISPATCH_FLOAT_TYPES(impl::cuda::knn_forward_cuda_kernel,
                         new_xyz.scalar_type(),
                         blocks,
                         threads,
                         stream,
                         b,
                         n,
                         m,
                         nsample,
                         xyz.data(),
                         new_xyz.data(),
                         static_cast<int *>(idx.data()),
                         dist2.data());

    return diopiSuccess;
}
