/**
 * @file assign_score_withk.cu
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <assert.h>
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
template <typename T>
__global__ void assign_score_withk_forward_cuda_kernel_diopi(const int B, const int N0, const int N1, const int M, const int K, const int O,
                                                             const int aggregate, const void* points_, const void* centers_, const void* scores_,
                                                             const int64_t* knn_idx, void* output_) {
    const T* points = static_cast<const T*>(points_);
    const T* centers = static_cast<const T*>(centers_);
    const T* scores = static_cast<const T*>(scores_);
    T* output = static_cast<T*>(output_);
    // ----- parallel loop for B, N1, K and O ---------
    CUDA_1D_KERNEL_LOOP(i, B * O * N1 * K) {
        // ------- loop for M ----------
        const int b = static_cast<int>(i / (O * N1 * K));
        const int o = static_cast<int>(i % (O * N1 * K) / (N1 * K));
        const int n = static_cast<int>(i % (N1 * K) / K);
        const int k = static_cast<int>(i % K);
        const int cn = static_cast<int>(knn_idx[b * K * N1 + n * K + 0]);  // The first neighbor is the center point
        const int kn = static_cast<int>(knn_idx[b * K * N1 + n * K + k]);
        if (kn >= N0 || kn < 0) {  // if index overflows, it is out of the neighborhood range
            return;
        }
        assert(b < B);
        assert(kn < N0);
        assert(cn < N0);
        assert(o < O);
        assert(n < N1);
        const int out_idx = b * N1 * O * K + o * N1 * K + n * K + k;
        T val = output[out_idx];
        for (int m = 0; m < M; m++) {
            val += points[b * N0 * M * O + kn * M * O + m * O + o] * scores[b * N1 * K * M + n * K * M + k * M + m] -
                   centers[b * N0 * M * O + cn * M * O + m * O + o] * scores[b * N1 * K * M + n * K * M + k * M + m];
        }
        output[out_idx] = val;
    }
}

template <typename T>
__global__ void assign_score_withk_points_backward_cuda_kernel_diopi(const int B, const int N0, const int N, const int M, const int K, const int O,
                                                                     const int aggregate, const void* grad_out_, const void* scores_, const int64_t* knn_idx,
                                                                     void* grad_points_, void* grad_centers_) {
    const T* grad_out = static_cast<const T*>(grad_out_);
    const T* scores = static_cast<const T*>(scores_);
    T* grad_points = static_cast<T*>(grad_points_);
    T* grad_centers = static_cast<T*>(grad_centers_);
    // ----- parallel loop for B, M, O ---------
    CUDA_1D_KERNEL_LOOP(i, B * M * O) {
        int b = static_cast<int>(i / (M * O));
        int m = static_cast<int>(i % (M * O) / O);
        int o = static_cast<int>(i % O);

        // ----- loop for N,K ---------
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                int kn = knn_idx[b * N * K + n * K + k];
                int cn = knn_idx[b * N * K + n * K + 0];
                if (kn >= N0 || kn < 0) {  // if index overflows, it is out of the
                                           // neighborhood range
                    continue;
                }
                atomicAdd(grad_points + b * N0 * M * O + kn * M * O + m * O + o,
                          scores[b * N * K * M + n * K * M + k * M + m] * grad_out[b * O * N * K + o * N * K + n * K + k]);
                atomicAdd(grad_centers + b * N0 * M * O + cn * M * O + m * O + o,
                          -scores[b * N * K * M + n * K * M + k * M + m] * grad_out[b * O * N * K + o * N * K + n * K + k]);
            }
        }
    }
}

template <typename T>
__global__ void assign_score_withk_scores_backward_cuda_kernel_diopi(const int B, const int N0, const int N, const int M, const int K, const int O,
                                                                     const int aggregate, const void* grad_out_, const void* points_, const void* centers_,
                                                                     const int64_t* knn_idx, void* grad_scores_) {
    const T* grad_out = static_cast<const T*>(grad_out_);
    const T* points = static_cast<const T*>(points_);
    const T* centers = static_cast<const T*>(centers_);
    T* grad_scores = static_cast<T*>(grad_scores_);
    // ----- parallel loop for B, N, K, M ---------
    CUDA_1D_KERNEL_LOOP(i, B * N * K * M) {
        const int b = static_cast<int>(i / (N * M * K));
        const int n = static_cast<int>(i % (N * M * K) / M / K);
        const int k = static_cast<int>(i % (M * K) / M);
        const int m = static_cast<int>(i % M);
        const int cn = knn_idx[b * N * K + n * K + 0];
        const int kn = knn_idx[b * N * K + n * K + k];
        if (kn >= N0 || kn < 0) {  // if index overflows, it is out of the neighborhood range
            return;
        }

        // -------------- loop for O ------------------------
        const int out_idx = b * N * K * M + n * K * M + k * M + m;
        T val = grad_scores[out_idx];
        for (int o = 0; o < O; o++) {
            val += (points[b * N0 * M * O + kn * M * O + m * O + o] - centers[b * N0 * M * O + cn * M * O + m * O + o]) *
                   grad_out[b * O * N * K + o * N * K + n * K + k];
        }
        grad_scores[out_idx] = val;
    }
}
}  // namespace cuda

}  // namespace impl

diopiError_t diopiAssignScoreWithk(diopiContextHandle_t ctx, diopiConstTensorHandle_t points_, diopiConstTensorHandle_t centers_,
                                   diopiConstTensorHandle_t scores_, diopiConstTensorHandle_t knn_idx_, diopiTensorHandle_t output_, int64_t B, int64_t N0,
                                   int64_t N1, int64_t M, int64_t K, int64_t O, int64_t aggregate) {
    auto points = impl::cuda::makeTensor(points_);
    auto centers = impl::cuda::makeTensor(centers_);
    auto scores = impl::cuda::makeTensor(scores_);
    auto knn_idx = impl::cuda::makeTensor(knn_idx_);
    auto output = impl::cuda::makeTensor(output_);

    // // at::cuda::CUDAGuard device_guard(points.device());
    auto stream = impl::cuda::getStream(ctx);

    dim3 blocks(GET_BLOCKS(B * O * N1 * K, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    DISPATCH_FLOAT_TYPES(impl::cuda::assign_score_withk_forward_cuda_kernel_diopi,
                         points.scalar_type(),
                         blocks,
                         threads,
                         stream,
                         B,
                         N0,
                         N1,
                         M,
                         K,
                         O,
                         aggregate,
                         points.data(),
                         centers.data(),
                         scores.data(),
                         static_cast<const int64_t*>(knn_idx.data()),
                         output.data());
    return diopiSuccess;
}

diopiError_t diopiAssignScoreWithkBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out_, diopiConstTensorHandle_t points_,
                                           diopiConstTensorHandle_t centers_, diopiConstTensorHandle_t scores_, diopiConstTensorHandle_t knn_idx_,
                                           diopiTensorHandle_t grad_points_, diopiTensorHandle_t grad_centers_, diopiTensorHandle_t grad_scores_, int64_t B,
                                           int64_t N0, int64_t N1, int64_t M, int64_t K, int64_t O, int64_t aggregate) {
    auto grad_out = impl::cuda::makeTensor(grad_out_);
    auto points = impl::cuda::makeTensor(points_);
    auto centers = impl::cuda::makeTensor(centers_);
    auto scores = impl::cuda::makeTensor(scores_);
    auto knn_idx = impl::cuda::makeTensor(knn_idx_);
    auto grad_points = impl::cuda::makeTensor(grad_points_);
    auto grad_centers = impl::cuda::makeTensor(grad_centers_);
    auto grad_scores = impl::cuda::makeTensor(grad_scores_);

    // // at::cuda::CUDAGuard device_guard(grad_out.device());
    auto stream = impl::cuda::getStream(ctx);

    dim3 blocks1(GET_BLOCKS(B * M * O, THREADS_PER_BLOCK));
    dim3 threads1(THREADS_PER_BLOCK);
    dim3 blocks2(GET_BLOCKS(B * N1 * K * M, THREADS_PER_BLOCK));
    dim3 threads2(THREADS_PER_BLOCK);

    DISPATCH_FLOAT_TYPES(impl::cuda::assign_score_withk_points_backward_cuda_kernel_diopi,
                         grad_out.scalar_type(),
                         blocks1,
                         threads1,
                         stream,
                         B,
                         N0,
                         N1,
                         M,
                         K,
                         O,
                         aggregate,
                         grad_out.data(),
                         scores.data(),
                         static_cast<const int64_t*>(knn_idx.data()),
                         grad_points.data(),
                         grad_centers.data());

    DISPATCH_FLOAT_TYPES(impl::cuda::assign_score_withk_scores_backward_cuda_kernel_diopi,
                         grad_out.scalar_type(),
                         blocks2,
                         threads2,
                         stream,
                         B,
                         N0,
                         N1,
                         M,
                         K,
                         O,
                         aggregate,
                         grad_out.data(),
                         points.data(),
                         centers.data(),
                         static_cast<const int64_t*>(knn_idx.data()),
                         grad_scores.data());

    return diopiSuccess;
}
