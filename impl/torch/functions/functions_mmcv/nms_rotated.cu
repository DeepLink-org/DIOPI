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
#include "box_iou_rotated_uils.hpp"
using namespace cuda::helper;
namespace mmcv {
namespace ops {

int const threadsPerBlock = sizeof(unsigned long long int) * 8;

__host__ __device__ inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

template <typename T>
__global__ void nms_rotated_kernel(const int n_boxes, const float iou_threshold, const T *boxes, unsigned long long *mask, const int multi_label) {
    const int row_size = min(n_boxes - blockIdx.y * threadsPerBlock, threadsPerBlock);
    const int col_size = min(n_boxes - blockIdx.x * threadsPerBlock, threadsPerBlock);
    // shared_memory: (x_center, y_center, width, height, angle_degrees) here.
    __shared__ T block_boxes[threadsPerBlock * 5];
    if (multi_label == 1) {
        if (threadIdx.x < col_size) {
            // boxes shape [N, 6]
            block_boxes[threadIdx.x * 5 + 0] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 6 + 0];
            block_boxes[threadIdx.x * 5 + 1] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 6 + 1];
            block_boxes[threadIdx.x * 5 + 2] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 6 + 2];
            block_boxes[threadIdx.x * 5 + 3] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 6 + 3];
            block_boxes[threadIdx.x * 5 + 4] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 6 + 4];
        }
        __syncthreads();

        if (threadIdx.x < row_size) {
            // get a specific box in current thread
            const int cur_box_idx = threadsPerBlock * blockIdx.y + threadIdx.x;
            const T *cur_box = boxes + cur_box_idx * 6;
            unsigned long long t = 0;
            int start = 0;
            if (blockIdx.y == blockIdx.x) {
                // avoid compare with self
                start = threadIdx.x + 1;
            }
            for (int i = start; i < col_size; i++) {
                // compare with other boxes, update t (by bit)
                if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5, 0) > iou_threshold) {
                    t |= 1ULL << i;
                }
            }
            const int block_num = divideUP(n_boxes, threadsPerBlock);
            mask[cur_box_idx * block_num + blockIdx.x] = t;
        }
    } else {
        if (threadIdx.x < col_size) {
            // boxes shape [N, 5]
            block_boxes[threadIdx.x * 5 + 0] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 5 + 0];
            block_boxes[threadIdx.x * 5 + 1] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 5 + 1];
            block_boxes[threadIdx.x * 5 + 2] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 5 + 2];
            block_boxes[threadIdx.x * 5 + 3] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 5 + 3];
            block_boxes[threadIdx.x * 5 + 4] = boxes[(threadsPerBlock * blockIdx.x + threadIdx.x) * 5 + 4];
        }
        __syncthreads();

        if (threadIdx.x < row_size) {
            const int cur_box_idx = threadsPerBlock * blockIdx.y + threadIdx.x;
            const T *cur_box = boxes + cur_box_idx * 5;
            unsigned long long t = 0;
            int start = 0;
            if (blockIdx.y == blockIdx.x) {
                start = threadIdx.x + 1;
            }
            for (int i = start; i < col_size; i++) {
                if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5, 0) > iou_threshold) {
                    t |= 1ULL << i;
                }
            }
            const int block_num = divideUP(n_boxes, threadsPerBlock);
            mask[cur_box_idx * block_num + blockIdx.x] = t;
        }
    }
}

Tensor NMSRotatedCUDAKernelLauncher(const Tensor boxes, const Tensor scores, const Tensor order_t, const Tensor boxes_sorted, float iou_threshold,
                                    const int multi_label) {
    // set current device to boxes's device
    at::cuda::CUDAGuard device_guard(boxes.device());
    int boxes_num = boxes.size(0);
    const int block_num = divideUP(boxes_num, threadsPerBlock);

    Tensor mask = at::empty({boxes_num * block_num}, boxes.options().dtype(at::kLong));

    dim3 blocks(block_num, block_num);
    dim3 threads(threadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // scalar_t is from boxes_sorted.scalar_type()
    // here AT_DISPATCH_FLOATING_TYPES_AND_HALF will choose the correct dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes_sorted.scalar_type(), "nms_rotated_kernel", [&] {
        // shared memory usage set to 0
        nms_rotated_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            boxes_num, iou_threshold, boxes_sorted.data_ptr<scalar_t>(), (unsigned long long *)mask.data_ptr<int64_t>(), multi_label);
    });

    Tensor mask_cpu = mask.to(at::kCPU);
    unsigned long long *mask_host = (unsigned long long *)mask_cpu.data_ptr<int64_t>();

    std::vector<unsigned long long> removed_blocks(block_num);
    memset(&removed_blocks[0], 0, sizeof(unsigned long long) * block_num);

    Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
    int64_t *keep_out = keep.data_ptr<int64_t>();
    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int block_id = i / threadsPerBlock;
        int tid_in_block = i % threadsPerBlock;

        if (!(removed_blocks[block_id] & (1ULL << tid_in_block))) {
            keep_out[num_to_keep++] = i;
            unsigned long long *p = mask_host + i * block_num;
            for (int j = block_id; j < block_num; j++) {
                removed_blocks[j] |= p[j];
            }
        }
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return order_t.index({keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(order_t.device(), keep.scalar_type())});
}

}  // namespace ops
}  // namespace mmcv
