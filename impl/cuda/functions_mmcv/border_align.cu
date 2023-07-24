/**
 * @file border_align.cu
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <float.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include "../cuda_helper.hpp"
#include "../helper.hpp"

namespace impl {

namespace cuda {
enum BorderMode { Top = 0, Left = 1, Bottom = 2, Right = 3 };

/*** Forward ***/
template <typename T>
__global__ void border_align_forward_cuda_kernel(const int nthreads, const void* input_, const void* boxes_, void* output_, int* argmax_idx, const int channels,
                                                 const int box_size, const int height, const int width, const int pool_size) {
    const T* input = static_cast<const T*>(input_);
    const T* boxes = static_cast<const T*>(boxes_);
    T* output = static_cast<T*>(output_);
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (batch_idx, c_idx, box_idx) is an element paralleled for computing
        // output, and `extreme_idx` is in range [0,3]
        int batch_idx, c_idx, box_idx, extreme_idx, maxidx, *offset_argmax_idx;
        const T *offset_box, *offset_input, *offset_box_x;
        T *offset_output, box_width, box_height, stride, x_stride, y_stride, x, y, val, maxval;

        extreme_idx = threadIdx.y;
        // shape (N, C, box_size, 4) for output
        batch_idx = index / channels / box_size;
        // shape (N, box_size, 4) for boxes
        box_idx = index % box_size + batch_idx * box_size;
        c_idx = (index / box_size) % channels;

        offset_box = boxes + box_idx * 4;
        box_width = *(offset_box + 2) - *offset_box;
        box_height = *(offset_box + 3) - *(offset_box + 1);
        offset_output = output + index * 4 + extreme_idx;
        offset_argmax_idx = argmax_idx + index * 4 + extreme_idx;
        // shape (N, 4C, h, w) for input.
        // [0,C) for top feature, [C,2C) for left feature,
        // [2C,3C) for bottom feature, [3C,4C) for right feature
        offset_input = input + (batch_idx * channels * 4 + extreme_idx * channels + c_idx) * height * width;

        // extreme_idx in [0,1] -> offset_box_x indexed at x1
        // extreme_idx in [2,3] -> offset_box_x indexed at x2
        offset_box_x = offset_box + extreme_idx / 2 * 2;

        // (x1,y1) or (x2,y2) for (x,y)
        x = *offset_box_x;
        y = *(offset_box_x + 1);

        switch (extreme_idx) {
            // top
            case BorderMode::Top:
                stride = box_width / pool_size;
                x_stride = stride;
                y_stride = 0;
                break;
            // left
            case BorderMode::Left:
                stride = box_height / pool_size;
                x_stride = 0;
                y_stride = stride;
                break;
            // bottom
            case BorderMode::Bottom:
                stride = box_width / pool_size;
                x_stride = -stride;
                y_stride = 0;
                break;
            // right
            case BorderMode::Right:
                stride = box_height / pool_size;
                x_stride = 0;
                y_stride = -stride;
                break;
        }

        // initialize maxval and maxidx with the start position (e.g. (x1,y1) or
        // (x2,y2))
        maxval = bilinear_interpolate(offset_input, height, width, y, x, index);
        maxidx = 0;

        // do max_pool along the border
        for (int i = 1; i <= pool_size; i++) {
            x += x_stride;
            y += y_stride;
            val = bilinear_interpolate(offset_input, height, width, y, x, index);
            if (val > maxval) {
                maxval = val;
                maxidx = i;
            }
        }

        // update output and argmax_idx
        *offset_output = maxval;
        *offset_argmax_idx = maxidx;
    }
}

/*** Backward ***/
template <typename T>
__global__ void border_align_backward_cuda_kernel(const int nthreads, const void* grad_output_, const void* boxes_, const int* argmax_idx, void* grad_input_,
                                                  const int channels, const int box_size, const int height, const int width, const int pool_size) {
    const T* grad_output = static_cast<const T*>(grad_output_);
    const T* boxes = static_cast<const T*>(boxes_);
    T* grad_input = static_cast<T*>(grad_input_);
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (batch_idx, c_idx, box_idx) is an element paralleled for computing
        // output, and `extreme_idx` is in range [0,3]
        int batch_idx, c_idx, box_idx, extreme_idx;
        const int* offset_argmax_idx;
        const T *offset_grad_output, *offset_box, *offset_box_x;
        T *offset_grad_input, box_width, box_height, stride, x_stride, y_stride, x, y;

        extreme_idx = threadIdx.y;
        batch_idx = index / channels / box_size;
        box_idx = index % box_size + batch_idx * box_size;
        c_idx = (index / box_size) % channels;

        offset_box = boxes + box_idx * 4;
        box_width = *(offset_box + 2) - *offset_box;
        box_height = *(offset_box + 3) - *(offset_box + 1);
        offset_grad_output = grad_output + index * 4 + extreme_idx;
        offset_argmax_idx = argmax_idx + index * 4 + extreme_idx;
        // [0,C) for top feature grad, [C,2C) for left feature grad,
        // [2C,3C) for bottom feature grad, [3C,4C) for right feature grad
        offset_grad_input = grad_input + (batch_idx * channels * 4 + extreme_idx * channels + c_idx) * height * width;

        // extreme_idx in [0,1] -> offset_box_x indexed at x1
        // extreme_idx in [2,3] -> offset_box_x indexed at x2
        offset_box_x = offset_box + extreme_idx / 2 * 2;

        switch (extreme_idx) {
            // top
            case BorderMode::Top:
                stride = box_width / pool_size;
                x_stride = stride;
                y_stride = 0;
                break;
            // left
            case BorderMode::Left:
                stride = box_height / pool_size;
                x_stride = 0;
                y_stride = stride;
                break;
            // bottom
            case BorderMode::Bottom:
                stride = box_width / pool_size;
                x_stride = -stride;
                y_stride = 0;
                break;
            // right
            case BorderMode::Right:
                stride = box_height / pool_size;
                x_stride = 0;
                y_stride = -stride;
                break;
        }

        // get position (x,y) which has maximum value during forward
        x = *offset_box_x;
        y = *(offset_box_x + 1);
        x += x_stride * (T)(*offset_argmax_idx);
        y += y_stride * (T)(*offset_argmax_idx);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high, index);

        // update grad_output
        atomicAdd(offset_grad_input + y_low * width + x_low, *offset_grad_output * w1);
        atomicAdd(offset_grad_input + y_low * width + x_high, *offset_grad_output * w2);
        atomicAdd(offset_grad_input + y_high * width + x_low, *offset_grad_output * w3);
        atomicAdd(offset_grad_input + y_high * width + x_high, *offset_grad_output * w4);
    }
}
}  // namespace cuda

}  // namespace impl

diopiError_t diopiBorderAlign(diopiContextHandle_t ctx, diopiConstTensorHandle_t input_, diopiConstTensorHandle_t boxes_, diopiTensorHandle_t output_,
                              diopiTensorHandle_t argmax_idx_, const int64_t pool_size) {
    auto input = impl::cuda::makeTensor(input_);
    auto boxes = impl::cuda::makeTensor(boxes_);
    auto output = impl::cuda::makeTensor(output_);
    auto argmax_idx = impl::cuda::makeTensor(argmax_idx_);
    // shape assertion
    assert(input.ndimension() == 4);
    assert(boxes.ndimension() == 3);

    int batch_size = input.size(0);
    int feat_channels = input.size(1);
    int channels = feat_channels / 4;
    int height = input.size(2);
    int width = input.size(3);
    // shape [N, box_size, 4] for boxes. (x1, y1, x2, y2) format
    int box_size = boxes.size(1);
    // shape [N, channels, box_size, 4] for output
    int nthreads = batch_size * channels * box_size;

    // at::cuda::CUDAGuard device_guard(input.device());
    auto stream = impl::cuda::getStream(ctx);
    dim3 block(128, 4);
    DISPATCH_FLOAT_TYPES(impl::cuda::border_align_forward_cuda_kernel,
                         input.scalar_type(),
                         GET_BLOCKS(nthreads),
                         block,
                         stream,
                         nthreads,
                         input.data(),
                         boxes.data(),
                         output.data(),
                         static_cast<int*>(argmax_idx.data()),
                         channels,
                         box_size,
                         height,
                         width,
                         pool_size);
    return diopiSuccess;
}

diopiError_t diopiBorderAlignBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_output_, diopiConstTensorHandle_t boxes_,
                                      diopiConstTensorHandle_t argmax_idx_, diopiTensorHandle_t grad_input_, const int64_t pool_size) {
    auto grad_output = impl::cuda::makeTensor(grad_output_);
    auto boxes = impl::cuda::makeTensor(boxes_);
    auto argmax_idx = impl::cuda::makeTensor(argmax_idx_);
    auto grad_input = impl::cuda::makeTensor(grad_input_);

    int batch_size = grad_input.size(0);
    int feat_channels = grad_input.size(1);
    int channels = feat_channels / 4;
    int height = grad_input.size(2);
    int width = grad_input.size(3);
    int box_size = boxes.size(1);
    int nthreads = batch_size * channels * box_size;

    // at::cuda::CUDAGuard device_guard(grad_output.device());
    auto stream = impl::cuda::getStream(ctx);
    dim3 block(128, 4);
    DISPATCH_FLOAT_TYPES(impl::cuda::border_align_backward_cuda_kernel,
                         grad_output.scalar_type(),
                         GET_BLOCKS(nthreads),
                         block,
                         stream,
                         nthreads,
                         grad_output.data(),
                         boxes.data(),
                         static_cast<const int*>(argmax_idx.data()),
                         grad_input.data(),
                         channels,
                         box_size,
                         height,
                         width,
                         pool_size);
    return diopiSuccess;
}
