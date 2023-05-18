/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../cuda_helpers.h"

namespace mmcv {
namespace ops {

using namespace at;

template <typename T>
__global__ void sigmoid_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* output, const T gamma, const T alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + expf(-input[index]));

    // (1 - p)**gamma * log(p)
    T term_p = pow(((T)1. - p), gamma) * log(max(p, (T)FLT_MIN));
    // p**gamma * log(1 - p)
    T term_n = pow(p, gamma) * log(max((T)1. - p, (T)FLT_MIN));

    output[index] = (T)0.;
    output[index] += -flag_p * alpha * term_p;
    output[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      output[index] *= weight[t];
    }
  }
}

template <typename T>
__global__ void sigmoid_focal_loss_backward_cuda_kernel(
    const int nthreads, const T* input, const int64_t* target, const T* weight,
    T* grad_input, const T gamma, const T alpha, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;

    int64_t t = target[n];
    T flag_p = (t == c);
    T flag_n = (t != c);

    // p = sigmoid(x) = 1. / 1. + expf(-x)
    T p = (T)1. / ((T)1. + exp(-input[index]));

    // (1 - p)**gamma * (1 - p - gamma*p*log(p))
    T term_p = pow(((T)1. - p), gamma) *
               ((T)1. - p - (gamma * p * log(max(p, (T)FLT_MIN))));
    // p**gamma * (gamma * (1 - p) * log(1 - p) - p)
    T term_n = pow(p, gamma) *
               (gamma * ((T)1. - p) * log(max((T)1. - p, (T)FLT_MIN)) - p);

    grad_input[index] = (T)0.;
    grad_input[index] += -flag_p * alpha * term_p;
    grad_input[index] += -flag_n * ((T)1. - alpha) * term_n;
    if (weight != NULL) {
      grad_input[index] *= weight[t];
    }
  }
}

template <typename T>
__global__ void softmax_focal_loss_forward_cuda_kernel(
    const int nthreads, const T* softmax, const int64_t* target,
    const T* weight, T* output, const T gamma, const T alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int64_t label = target[index];
    T pred = softmax[index * num_classes + label];

    if (label >= 0) {
      output[index] =
          -alpha * pow((T)1. - pred, gamma) * log(max(pred, (T)FLT_MIN));
    } else {
      output[index] = 0;
    }
    if (weight != NULL) {
      output[index] *= weight[label];
    }
  }
}

template <typename T>
__global__ void softmax_focal_loss_backward_cuda1_kernel(
    const int nthreads, const T* softmax, const int64_t* target,
    const T* weight, T* buff, const T gamma, const T alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int64_t label = target[index];
    T pred = softmax[index * num_classes + label];

    if (label >= 0) {
      buff[index] = alpha * (-pow((T)1. - pred, gamma) +
                             gamma * pow((T)1. - pred, gamma - 1) * pred *
                                 log(max(pred, (T)FLT_MIN)));
    } else {
      buff[index] = 0;
    }
    if (weight != NULL) {
      buff[index] *= weight[label];
    }
  }
}

template <typename T>
__global__ void softmax_focal_loss_backward_cuda2_kernel(
    const int nthreads, const T* softmax, const int64_t* target, const T* buff,
    T* grad_input, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index / num_classes;
    int c = index % num_classes;
    int64_t label = target[n];

    if (label >= 0) {
      T flag = (label == c ? (T)1. : (T)0.);
      grad_input[index] = buff[n] * (flag - softmax[index]);
    } else {
      grad_input[index] = 0;
    }
  }
}

void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = input.size(1);
  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_forward_cuda_kernel", [&] {
        sigmoid_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target,
                                                Tensor weight,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int output_size = grad_input.numel();
  int num_classes = input.size(1);

  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "sigmoid_focal_loss_backward_cuda_kernel", [&] {
        sigmoid_focal_loss_backward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha) {
  int output_size = output.numel();
  int num_classes = softmax.size(1);

  AT_ASSERTM(target.max().item<int64_t>() <= (int64_t)num_classes,
             "target label should smaller or equal than num classes");
  at::cuda::CUDAGuard device_guard(softmax.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      softmax.scalar_type(), "softmax_focal_loss_forward_cuda_kernel", [&] {
        softmax_focal_loss_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor softmax, Tensor target,
                                                Tensor weight, Tensor buff,
                                                Tensor grad_input,
                                                const float gamma,
                                                const float alpha) {
  int num_classes = softmax.size(1);

  int output_size = buff.numel();
  at::cuda::CUDAGuard device_guard(grad_input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(),
      "softmax_focal_loss_backward_cuda1_"
      "kernel",
      [&] {
        softmax_focal_loss_backward_cuda1_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), weight.data_ptr<scalar_t>(),
                buff.data_ptr<scalar_t>(), gamma, alpha, num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());

  output_size = grad_input.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_input.scalar_type(),
      "softmax_focal_loss_backward_cuda2_"
      "kernel",
      [&] {
        softmax_focal_loss_backward_cuda2_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, softmax.data_ptr<scalar_t>(),
                target.data_ptr<int64_t>(), buff.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(), num_classes);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops
}  // namespace mmcv
