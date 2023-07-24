/**
 * @file deform_roi_pool.cu
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
__global__ void deform_roi_pool_forward_cuda_kernel(const int nthreads, const void* input_, const void* rois_, const void* offset_, void* output_,
                                                    const int pooled_height, const int pooled_width, const T spatial_scale, const int sampling_ratio,
                                                    const T gamma, const int channels, const int height, const int width) {
    const T* input = static_cast<const T*>(input_);
    const T* rois = static_cast<const T*>(rois_);
    const T* offset = static_cast<const T*>(offset_);
    T* output = static_cast<T*>(output_);
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_rois[1] * spatial_scale - 0.5;
        T roi_start_h = offset_rois[2] * spatial_scale - 0.5;
        T roi_end_w = offset_rois[3] * spatial_scale - 0.5;
        T roi_end_h = offset_rois[4] * spatial_scale - 0.5;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        const T* offset_input = input + (roi_batch_ind * channels + c) * height * width;

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_height / pooled_height));
        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_width / pooled_width));

        // Compute roi offset
        if (offset != NULL) {
            const T* offset_cur_w = offset + n * pooled_width * pooled_height * 2 + ph * pooled_width + pw;
            T offset_roi_w = gamma * roi_width * offset_cur_w[0];
            T offset_roi_h = gamma * roi_height * offset_cur_w[pooled_width * pooled_height];
            roi_start_w += offset_roi_w;
            roi_start_h += offset_roi_h;
        }

        // We do average pooling inside a bin
        const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
        T output_val = 0.;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
                T val = bilinear_interpolate(offset_input, height, width, y, x, index);
                output_val += val;
            }
        }
        output[index] = output_val / count;
    }
}

template <typename T>
__global__ void deform_roi_pool_backward_cuda_kernel(const int nthreads, const void* grad_output_, const void* input_, const void* rois_, const void* offset_,
                                                     void* grad_input_, void* grad_offset_, const int pooled_height, const int pooled_width,
                                                     const T spatial_scale, const int sampling_ratio, const T gamma, const int channels, const int height,
                                                     const int width) {
    const T* grad_output = static_cast<const T*>(grad_output_);
    const T* input = static_cast<const T*>(input_);
    const T* rois = static_cast<const T*>(rois_);
    const T* offset = static_cast<const T*>(offset_);
    T* grad_input = static_cast<T*>(grad_input_);
    T* grad_offset = static_cast<T*>(grad_offset_);
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];
        const T* offset_input = input + ((roi_batch_ind * channels + c) * height * width);
        T* offset_grad_input = grad_input + ((roi_batch_ind * channels + c) * height * width);

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_rois[1] * spatial_scale - 0.5;
        T roi_start_h = offset_rois[2] * spatial_scale - 0.5;
        T roi_end_w = offset_rois[3] * spatial_scale - 0.5;
        T roi_end_h = offset_rois[4] * spatial_scale - 0.5;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_height / pooled_height));
        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_width / pooled_width));

        // Compute roi offset
        if (offset != NULL) {
            const T* offset_cur_w = offset + n * pooled_width * pooled_height * 2 + ph * pooled_width + pw;
            T offset_roi_w = gamma * roi_width * offset_cur_w[0];
            T offset_roi_h = gamma * roi_height * offset_cur_w[pooled_width * pooled_height];
            roi_start_w += offset_roi_w;
            roi_start_h += offset_roi_h;
        }

        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
        const T grad_output_this_bin = grad_output[index] / count;

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                T w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;
                bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high, index);

                if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                    atomicAdd(offset_grad_input + y_low * width + x_low, grad_output_this_bin * w1);
                    atomicAdd(offset_grad_input + y_low * width + x_high, grad_output_this_bin * w2);
                    atomicAdd(offset_grad_input + y_high * width + x_low, grad_output_this_bin * w3);
                    atomicAdd(offset_grad_input + y_high * width + x_high, grad_output_this_bin * w4);
                    if (offset != NULL) {
                        T input_00 = offset_input[y_low * width + x_low];
                        T input_10 = offset_input[y_low * width + x_high];
                        T input_01 = offset_input[y_high * width + x_low];
                        T input_11 = offset_input[y_high * width + x_high];
                        T ogx = gamma * roi_width * grad_output_this_bin *
                                (input_11 * (y - y_low) + input_10 * (y_high - y) + input_01 * (y_low - y) + input_00 * (y - y_high));
                        T ogy = gamma * roi_height * grad_output_this_bin *
                                (input_11 * (x - x_low) + input_01 * (x_high - x) + input_10 * (x_low - x) + input_00 * (x - x_high));
                        atomicAdd(grad_offset + n * pooled_width * pooled_height * 2 + ph * pooled_width + pw, ogx);
                        atomicAdd(grad_offset + n * pooled_width * pooled_height * 2 + pooled_width * pooled_height + ph * pooled_width + pw, ogy);
                    }
                }
            }
        }
    }
}
}  // namespace cuda

}  // namespace impl

diopiError_t diopiDeformRoiPool(diopiContextHandle_t ctx, diopiTensorHandle_t input_, diopiTensorHandle_t rois_, diopiTensorHandle_t offset_,
                                diopiTensorHandle_t output_, int64_t pooled_height, int64_t pooled_width, float spatial_scale, int64_t sampling_ratio,
                                float gamma) {
    auto input = impl::cuda::makeTensor(input_);
    auto rois = impl::cuda::makeTensor(rois_);
    auto offset = impl::cuda::makeTensor(offset_);
    auto output = impl::cuda::makeTensor(output_);
    int output_size = output.numel();
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // at::cuda::CUDAGuard device_guard(input.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::deform_roi_pool_forward_cuda_kernel,
                         input.scalar_type(),
                         GET_BLOCKS(output_size),
                         THREADS_PER_BLOCK,
                         stream,
                         output_size,
                         input.data(),
                         rois.data(),
                         offset.data(),
                         output.data(),
                         pooled_height,
                         pooled_width,
                         spatial_scale,
                         sampling_ratio,
                         gamma,
                         channels,
                         height,
                         width);

    return diopiSuccess;
}

diopiError_t diopiDeformRoiPoolBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output_, diopiTensorHandle_t input_, diopiTensorHandle_t rois_,
                                        diopiTensorHandle_t offset_, diopiTensorHandle_t grad_input_, diopiTensorHandle_t grad_offset_, int64_t pooled_height,
                                        int64_t pooled_width, float spatial_scale, int64_t sampling_ratio, float gamma) {
    auto grad_output = impl::cuda::makeTensor(grad_output_);
    auto input = impl::cuda::makeTensor(input_);
    auto rois = impl::cuda::makeTensor(rois_);
    auto offset = impl::cuda::makeTensor(offset_);
    auto grad_input = impl::cuda::makeTensor(grad_input_);
    auto grad_offset = impl::cuda::makeTensor(grad_offset_);

    int output_size = grad_output.numel();
    int channels = grad_input.size(1);
    int height = grad_input.size(2);
    int width = grad_input.size(3);

    // at::cuda::CUDAGuard device_guard(grad_output.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::deform_roi_pool_backward_cuda_kernel,
                         grad_output.scalar_type(),
                         GET_BLOCKS(output_size),
                         THREADS_PER_BLOCK,
                         stream,
                         output_size,
                         grad_output.data(),
                         input.data(),
                         rois.data(),
                         offset.data(),
                         grad_input.data(),
                         grad_offset.data(),
                         pooled_height,
                         pooled_width,
                         spatial_scale,
                         sampling_ratio,
                         gamma,
                         channels,
                         height,
                         width);
    return diopiSuccess;
}
