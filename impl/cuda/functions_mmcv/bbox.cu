/**
 * @file bbox.cu
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
template <typename T>
__device__ __forceinline__ void load_bbox(const T* bbox, const int base, T& x1, T& y1, T& x2, T& y2) {
    x1 = bbox[base];
    y1 = bbox[base + 1];
    x2 = bbox[base + 2];
    y2 = bbox[base + 3];
}

template <>
__device__ __forceinline__ void load_bbox<float>(const float* bbox, const int base, float& x1, float& y1, float& x2, float& y2) {
    const float4 bbox_offset = reinterpret_cast<const float4*>(bbox + base)[0];
    x1 = bbox_offset.x;
    y1 = bbox_offset.y;
    x2 = bbox_offset.z;
    y2 = bbox_offset.w;
}

template <typename T>
__global__ void bbox_overlaps_cuda_kernel(const void* bbox1_, const void* bbox2_, void* ious_, const int num_bbox1, const int num_bbox2, const int mode,
                                          const bool aligned, const int offset) {
    const T* bbox1 = static_cast<const T*>(bbox1_);
    const T* bbox2 = static_cast<const T*>(bbox2_);
    T* ious = static_cast<T*>(ious_);
    if (aligned) {
        CUDA_1D_KERNEL_LOOP(index, num_bbox1) {
            const int b1 = index;
            const int b2 = index;

            const int base1 = b1 << 2;  // b1 * 4
            T b1_x1, b1_y1, b1_x2, b1_y2;
            load_bbox<T>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
            const T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

            const int base2 = b2 << 2;  // b2 * 4
            T b2_x1, b2_y1, b2_x2, b2_y2;
            load_bbox<T>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
            const T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

            const T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
            const T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
            const T width = fmaxf(right - left + offset, 0.f);
            const T height = fmaxf(bottom - top + offset, 0.f);
            const T interS = width * height;

            const T baseS = fmaxf(mode == 0 ? b1_area + b2_area - interS : b1_area, T(offset));
            ious[index] = interS / baseS;
        }
    } else {
        CUDA_1D_KERNEL_LOOP(index, num_bbox1 * num_bbox2) {
            const int b1 = index / num_bbox2;
            const int b2 = index % num_bbox2;

            const int base1 = b1 << 2;  // b1 * 4
            T b1_x1, b1_y1, b1_x2, b1_y2;
            load_bbox<T>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
            const T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

            const int base2 = b2 << 2;  // b2 * 4
            T b2_x1, b2_y1, b2_x2, b2_y2;
            load_bbox<T>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
            const T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

            const T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
            const T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
            const T width = fmaxf(right - left + offset, 0.f);
            const T height = fmaxf(bottom - top + offset, 0.f);
            const T interS = width * height;

            const T baseS = fmaxf(mode == 0 ? b1_area + b2_area - interS : b1_area, T(offset));
            ious[index] = interS / baseS;
        }
    }
}
}  // namespace cuda

}  // namespace impl

diopiError_t diopiBboxOverlaps(diopiContextHandle_t ctx, diopiConstTensorHandle_t bboxes1_, diopiConstTensorHandle_t bboxes2_, diopiTensorHandle_t ious_,
                               const int64_t mode, const bool aligned, const int64_t offset) {
    auto bboxes1 = impl::cuda::makeTensor(bboxes1_);
    auto bboxes2 = impl::cuda::makeTensor(bboxes2_);
    auto ious = impl::cuda::makeTensor(ious_);
    int output_size = ious.numel();
    int num_bbox1 = bboxes1.size(0);
    int num_bbox2 = bboxes2.size(0);

    // // at::cuda::CUDAGuard device_guard(bboxes1.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::bbox_overlaps_cuda_kernel,
                         bboxes1.scalar_type(),
                         GET_BLOCKS(output_size),
                         THREADS_PER_BLOCK,
                         stream,
                         bboxes1.data(),
                         bboxes2.data(),
                         ious.data(),
                         num_bbox1,
                         num_bbox2,
                         mode,
                         aligned,
                         offset);
    return diopiSuccess;
}
