/**
 * @file active_rotated_filter.cu
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

template <typename scalar_t>
__global__ void active_rotated_filter_forward_cuda_kernel_diopi(const int nthreads, const void* weight_data_, const int* indices_data,
                                                                const int num_input_planes, const int num_output_planes, const int num_orientations,
                                                                const int num_rotations, const int nEntry, void* output_data_) {
    const scalar_t* weight_data = static_cast<const scalar_t*>(weight_data_);
    scalar_t* output_data = static_cast<scalar_t*>(output_data_);
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int l = index % nEntry;
        int j = (index / nEntry) % num_input_planes;
        int i = index / nEntry / num_input_planes;
        int k;
        scalar_t val = *(weight_data + index);
        for (k = 0; k < num_rotations; k++) {
            int idx = static_cast<int>(*(indices_data + l * num_rotations + k)) - 1;
            scalar_t* target = output_data + i * (num_rotations * num_input_planes * nEntry) + k * (num_input_planes * nEntry) + j * (nEntry) + idx;
            *target = val;
        }
    }
}

template <typename scalar_t>
__global__ void active_rotated_filter_backward_cuda_kernel_diopi(const int nthreads, const void* gradWeight_data_, const int* indices_data,
                                                                 const int num_input_planes, const int num_output_planes, const int num_orientations,
                                                                 const int num_rotations, const int nEntry, void* weight_data_) {
    const scalar_t* gradWeight_data = static_cast<const scalar_t*>(gradWeight_data_);
    scalar_t* weight_data = static_cast<scalar_t*>(weight_data_);
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int l = index % nEntry;
        int j = (index / nEntry) % num_input_planes;
        int i = index / nEntry / num_input_planes;
        int k;
        scalar_t* val = weight_data + index;
        *val = 0;
        scalar_t tmp = 0;
        for (k = 0; k < num_rotations; k++) {
            int idx = static_cast<int>(*(indices_data + l * num_rotations + k)) - 1;
            scalar_t target = *(gradWeight_data + i * (num_rotations * num_input_planes * nEntry) + k * (num_input_planes * nEntry) + j * (nEntry) + idx);
            tmp = tmp + target;
        }
        *val = tmp;
    }
}

}  // namespace cuda

}  // namespace impl

extern "C" diopiError_t diopiActiveRotatedFilter(diopiContextHandle_t ctx, diopiConstTensorHandle_t input_, diopiConstTensorHandle_t indices_,
                                                 diopiTensorHandle_t output_) {
    auto input = impl::cuda::makeTensor(input_);
    auto indices = impl::cuda::makeTensor(indices_);
    auto output = impl::cuda::makeTensor(output_);

    int num_output_planes = input.size(0);
    int num_input_planes = input.size(1);
    int num_orientations = input.size(2);
    int kH = input.size(3);
    int kW = input.size(4);
    int num_rotations = indices.size(3);
    int nEntry = num_orientations * kH * kW;
    int output_size = input.numel();

    // // at::cuda::CUDAGuard device_guard(input.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::active_rotated_filter_forward_cuda_kernel_diopi,
                         input.dtype(),
                         GET_BLOCKS(output_size),
                         THREADS_PER_BLOCK,
                         stream,
                         output_size,
                         input.data(),
                         static_cast<const int*>(indices.data()),
                         num_input_planes,
                         num_output_planes,
                         num_orientations,
                         num_rotations,
                         nEntry,
                         output.data());
    return diopiSuccess;
}

extern "C" diopiError_t diopiActiveRotatedFilterBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t grad_out_, diopiConstTensorHandle_t indices_,
                                                         diopiTensorHandle_t grad_in_) {
    auto grad_out = impl::cuda::makeTensor(grad_out_);
    auto indices = impl::cuda::makeTensor(indices_);
    auto grad_in = impl::cuda::makeTensor(grad_in_);

    int num_orientations = indices.size(0);
    int kH = indices.size(1);
    int kW = indices.size(2);
    int num_rotations = indices.size(3);
    int num_output_planes = grad_out.size(0) / num_rotations;
    int num_input_planes = grad_out.size(1) / num_orientations;
    int nEntry = num_orientations * kH * kW;
    int output_size = grad_in.numel();

    // // at::cuda::CUDAGuard device_guard(indices.device());
    auto stream = impl::cuda::getStream(ctx);
    DISPATCH_FLOAT_TYPES(impl::cuda::active_rotated_filter_backward_cuda_kernel_diopi,
                         grad_out.scalar_type(),
                         GET_BLOCKS(output_size),
                         THREADS_PER_BLOCK,
                         stream,
                         output_size,
                         grad_out.data(),
                         static_cast<const int*>(indices.data()),
                         num_input_planes,
                         num_output_planes,
                         num_orientations,
                         num_rotations,
                         nEntry,
                         grad_in.data());
    return diopiSuccess;
}
