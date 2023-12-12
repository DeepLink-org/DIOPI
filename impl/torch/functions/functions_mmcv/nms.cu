/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <torch/library.h>

#include "cuda_helpers.h"
using namespace cuda::helper;
namespace mmcv {
namespace ops {

using namespace at;

int const threadsPerBlock = sizeof(unsigned long long int) * 8;

__device__ inline bool devIoU(float const *const a, float const *const b,
                              const int offset, const float threshold) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + offset, 0.f),
        height = fmaxf(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS > threshold * (Sa + Sb - interS);
}

__global__ static void nms_cuda(const int n_boxes, const float iou_threshold,
                                const int offset, const float *dev_boxes,
                                unsigned long long *dev_mask) {
  int blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  CUDA_2D_KERNEL_BLOCK_LOOP(col_start, blocks, row_start, blocks) {
    const int tid = threadIdx.x;

    if (row_start > col_start) return;

    const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 4];
    if (tid < col_size) {
      block_boxes[tid * 4 + 0] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
      block_boxes[tid * 4 + 1] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
      block_boxes[tid * 4 + 2] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
      block_boxes[tid * 4 + 3] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
    }
    __syncthreads();

    if (tid < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + tid;
      const float *cur_box = dev_boxes + cur_box_idx * 4;
      int i = 0;
      unsigned long long int t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = tid + 1;
      }
      for (i = start; i < col_size; i++) {
        if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
          t |= 1ULL << i;
        }
      }
      dev_mask[cur_box_idx * gridDim.y + col_start] = t;
    }
  }
}

__global__ static void gather_keep_from_mask(bool *keep,
                                             const unsigned long long *dev_mask,
                                             const int n_boxes) {
  const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  const int tid = threadIdx.x;

  // mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // initialize removed.
  for (int i = tid; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int nblock = 0; nblock < col_blocks; ++nblock) {
    auto removed_val = removed[nblock];
    __syncthreads();
    const int i_offset = nblock * threadsPerBlock;
#pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; ++inblock) {
      const int i = i_offset + inblock;
      if (i >= n_boxes) break;
      // select a candidate, check if it should kept.
      if (!(removed_val & (1ULL << inblock))) {
        if (tid == 0) {
          // mark the output.
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;
        // remove all bboxes which overlap the candidate.
        for (int j = tid; j < col_blocks; j += blockDim.x) {
          if (j >= nblock) removed[j] |= p[j];
        }
        __syncthreads();
        removed_val = removed[nblock];
      }
    }
  }
}

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold,
                             int offset) {
  at::cuda::CUDAGuard device_guard(boxes.device());

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto order_t = std::get<1>(scores.sort(0, /*descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);
  const int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;
  const int col_blocks_alloc = GET_BLOCKS(boxes_num, threadsPerBlock);
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  dim3 blocks(col_blocks_alloc, col_blocks_alloc);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  nms_cuda<<<blocks, threads, 0, stream>>>(
      boxes_num, iou_threshold, offset, boxes_sorted.data_ptr<float>(),
      (unsigned long long *)mask.data_ptr<int64_t>());

  // Filter the boxes which should be kept.
  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(at::kCUDA));
  gather_keep_from_mask<<<1, min(col_blocks, THREADS_PER_BLOCK),
                          col_blocks * sizeof(unsigned long long), stream>>>(
      keep_t.data_ptr<bool>(), (unsigned long long *)mask.data_ptr<int64_t>(),
      boxes_num);
  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.masked_select(keep_t);
}

}  // namespace ops
}  // namespace mmcv
