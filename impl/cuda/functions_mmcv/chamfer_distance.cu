/**
 * @file chamfer_distance.cu
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
#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144

template <typename scalar_t>
__global__ void chamfer_distance_forward_cuda_kernel_diopi(int b, int n, const void* xyz, int m, const void* xyz2, void* result, int* result_i) {
    __shared__ scalar_t buf[MAX_SHARED_SCALAR_T];
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int k2 = 0; k2 < m; k2 += THREADS_PER_BLOCK) {
            int end_k = min(m, k2 + THREADS_PER_BLOCK) - k2;
            const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
            for (int j = threadIdx.x; j < end_k * 2; j += blockDim.x) {
                buf[j] = xyz2_[(i * m + k2) * 2 + j];
            }
            __syncthreads();
            const scalar_t* xyz_ = static_cast<const scalar_t*>(xyz);
            for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
                scalar_t x1 = xyz_[(i * n + j) * 2 + 0];
                scalar_t y1 = xyz_[(i * n + j) * 2 + 1];
                int best_i = 0;
                scalar_t best = 1e10;
                int end_ka = end_k & (~2);
                if (end_ka == THREADS_PER_BLOCK) {
                    for (int k = 0; k < THREADS_PER_BLOCK; k += 4) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            scalar_t x2 = buf[(k + j) * 2] - x1;
                            scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
                            scalar_t d = x2 * x2 + y2 * y2;
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + j;
                            }
                        }
                    }
                } else {
                    for (int k = 0; k < end_ka; k += 4) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            scalar_t x2 = buf[(k + j) * 2] - x1;
                            scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
                            scalar_t d = x2 * x2 + y2 * y2;
                            if (d < best) {
                                best = d;
                                best_i = k + k2 + j;
                            }
                        }
                    }
                }
                for (int k = end_ka; k < end_k; k++) {
                    scalar_t x2 = buf[k * 2 + 0] - x1;
                    scalar_t y2 = buf[k * 2 + 1] - y1;
                    scalar_t d = x2 * x2 + y2 * y2;
                    if (k == 0 || d < best) {
                        best = d;
                        best_i = k + k2;
                    }
                }
                scalar_t* result_ = static_cast<scalar_t*>(result);
                if (k2 == 0 || result_[(i * n + j)] > best) {
                    result_[(i * n + j)] = best;
                    result_i[(i * n + j)] = best_i;
                }
            }
            __syncthreads();
        }
    }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel_diopi(int b, int n, const void* xyz1, int m, const void* xyz2, const void* grad_dist1, const int* idx1,
                                                            void* grad_xyz1, void* grad_xyz2) {
    const scalar_t* xyz1_ = static_cast<const scalar_t*>(xyz1);
    const scalar_t* xyz2_ = static_cast<const scalar_t*>(xyz2);
    const scalar_t* grad_dist1_ = static_cast<const scalar_t*>(grad_dist1);
    scalar_t* grad_xyz1_ = static_cast<scalar_t*>(grad_xyz1);
    scalar_t* grad_xyz2_ = static_cast<scalar_t*>(grad_xyz2);
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
            scalar_t x1 = xyz1_[(i * n + j) * 2 + 0];
            scalar_t y1 = xyz1_[(i * n + j) * 2 + 1];
            int j2 = idx1[i * n + j];
            scalar_t x2 = xyz2_[(i * m + j2) * 2 + 0];
            scalar_t y2 = xyz2_[(i * m + j2) * 2 + 1];
            scalar_t g = grad_dist1_[i * n + j] * 2;
            atomicAdd(&(grad_xyz1_[(i * n + j) * 2 + 0]), g * (x1 - x2));
            atomicAdd(&(grad_xyz1_[(i * n + j) * 2 + 1]), g * (y1 - y2));
            atomicAdd(&(grad_xyz2_[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
            atomicAdd(&(grad_xyz2_[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
        }
    }
}
}  // namespace cuda

}  // namespace impl

extern "C" diopiError_t diopiChamferDistance(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in, diopiConstTensorHandle_t xyz2_in,
                                             diopiTensorHandle_t dist1_out, diopiTensorHandle_t dist2_out, diopiTensorHandle_t idx1_out,
                                             diopiTensorHandle_t idx2_out) {
    auto xyz1 = impl::cuda::makeTensor(xyz1_in);
    auto xyz2 = impl::cuda::makeTensor(xyz2_in);
    auto dist1 = impl::cuda::makeTensor(dist1_out);
    auto dist2 = impl::cuda::makeTensor(dist2_out);
    auto idx1 = impl::cuda::makeTensor(idx1_out);
    auto idx2 = impl::cuda::makeTensor(idx2_out);
    int batch_size = xyz1.size(0);
    int n = xyz1.size(1);
    int m = xyz2.size(1);
    // at::cuda::CUDAGuard device_guard(xyz1.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::chamfer_distance_forward_cuda_kernel_diopi,
                         xyz1.dtype(),
                         GET_BLOCKS(batch_size * n),
                         THREADS_PER_BLOCK,
                         stream,
                         batch_size,
                         n,
                         xyz1.data(),
                         m,
                         xyz2.data(),
                         dist1.data(),
                         static_cast<int*>(idx1.data()));
    DISPATCH_FLOAT_TYPES(impl::cuda::chamfer_distance_forward_cuda_kernel_diopi,
                         xyz1.dtype(),
                         GET_BLOCKS(batch_size * m),
                         THREADS_PER_BLOCK,
                         stream,
                         batch_size,
                         m,
                         xyz2.data(),
                         n,
                         xyz1.data(),
                         dist2.data(),
                         static_cast<int*>(idx2.data()));
    return diopiSuccess;
}

extern "C" diopiError_t diopiChamferDistanceBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t xyz1_in, diopiConstTensorHandle_t xyz2_in,
                                                     diopiConstTensorHandle_t idx1_in, diopiConstTensorHandle_t idx2_in, diopiConstTensorHandle_t grad_dist1_in,
                                                     diopiConstTensorHandle_t grad_dist2_in, diopiTensorHandle_t grad_xyz1_out,
                                                     diopiTensorHandle_t grad_xyz2_out) {
    auto xyz1 = impl::cuda::makeTensor(xyz1_in);
    auto xyz2 = impl::cuda::makeTensor(xyz2_in);
    auto idx1 = impl::cuda::makeTensor(idx1_in);
    auto idx2 = impl::cuda::makeTensor(idx2_in);
    auto grad_dist1 = impl::cuda::makeTensor(grad_dist1_in);
    auto grad_dist2 = impl::cuda::makeTensor(grad_dist2_in);
    auto grad_xyz1 = impl::cuda::makeTensor(grad_xyz1_out);
    auto grad_xyz2 = impl::cuda::makeTensor(grad_xyz2_out);
    int batch_size = xyz1.size(0);
    int n = xyz1.size(1);
    int m = xyz2.size(1);
    // at::cuda::CUDAGuard device_guard(xyz1.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::chamfer_distance_backward_cuda_kernel_diopi,
                         xyz1.dtype(),
                         GET_BLOCKS(batch_size * n),
                         THREADS_PER_BLOCK / 2,
                         stream,
                         batch_size,
                         m,
                         xyz1.data(),
                         n,
                         xyz2.data(),
                         grad_dist1.data(),
                         static_cast<const int*>(idx1.data()),
                         grad_xyz1.data(),
                         grad_xyz2.data());
    DISPATCH_FLOAT_TYPES(impl::cuda::chamfer_distance_backward_cuda_kernel_diopi,
                         xyz1.dtype(),
                         GET_BLOCKS(batch_size * m),
                         THREADS_PER_BLOCK / 2,
                         stream,
                         batch_size,
                         n,
                         xyz2.data(),
                         m,
                         xyz1.data(),
                         grad_dist2.data(),
                         static_cast<const int*>(idx2.data()),
                         grad_xyz2.data(),
                         grad_xyz1.data());
    return diopiSuccess;
}
