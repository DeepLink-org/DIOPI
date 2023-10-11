#include <ATen/native/TensorIterator.h>

#include <ATen/native/cuda/Loops.cuh>

#include "../cuda_helpers.h"

#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
switch(TYPEIN)							\
{									\
case at::ScalarType::Double:						\
    {									\
using scalar_t_in = double;					\
switch(TYPEOUT)							\
    {								\
    case at::ScalarType::Double:					\
    {								\
        using scalar_t_out = double;				\
        __VA_ARGS__;						\
        break;							\
    }								\
    case at::ScalarType::Float:					\
    {								\
        using scalar_t_out = float;				\
        __VA_ARGS__;						\
        break;							\
    }								\
    case at::ScalarType::Half:					\
    {								\
        using scalar_t_out = at::Half;				\
        __VA_ARGS__;						\
        break;							\
    }								\
    case at::ScalarType::BFloat16:				\
    {								\
        using scalar_t_out = at::BFloat16;			\
        __VA_ARGS__;						\
        break;							\
    }								\
    default:							\
    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
    }								\
break;								\
    }									\
case at::ScalarType::Float:						\
    {									\
using scalar_t_in = float;					\
switch(TYPEOUT)							\
    {								\
    case at::ScalarType::Float:					\
    {								\
        using scalar_t_out = float;				\
        __VA_ARGS__;						\
        break;							\
    }								\
    case at::ScalarType::Half:					\
    {								\
        using scalar_t_out = at::Half;				\
        __VA_ARGS__;						\
        break;							\
    }								\
    case at::ScalarType::BFloat16:				\
    {								\
        using scalar_t_out = at::BFloat16;			\
        __VA_ARGS__;						\
        break;							\
    }								\
    default:							\
    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
    }								\
break;								\
    }									\
case at::ScalarType::Half:						\
    {									\
using scalar_t_in = at::Half;					\
using scalar_t_out = at::Half;					\
__VA_ARGS__;							\
break;								\
    }									\
case at::ScalarType::BFloat16:					\
    {									\
using scalar_t_in = at::BFloat16;				\
using scalar_t_out = at::BFloat16;				\
__VA_ARGS__;							\
break;								\
    }									\
default:								\
    AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");	\
}







using namespace cuda::helper;
namespace ext {
namespace ops {

// using namespace at;

// forward

template<typename T, typename U, typename V> __device__
void cuApplyLayerNorm_(
  V* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma,
  const V* __restrict__ beta,
  bool rms_only
  )
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1=blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,i1,mu,sigma2,buf,rms_only);

    const T* lvals = vals + i1*n2;
    V* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }

      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template<typename T, typename U, typename V=T> __global__
void cuApplyRMSNorm(
  V* __restrict__ output_vals,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const V* __restrict__ gamma)
{
  cuApplyLayerNorm_<T, U, V>(output_vals, NULL, invvar, vals, n1, n2, epsilon, gamma, NULL, true);
}

template<typename T, typename U, typename V=T>
void HostApplyRMSNorm(
    V* output,
    U* invvar,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const V* gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32,4,1);
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
    int nshared =
        threads.y > 1 ?
            threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
            0;
    cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}



void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), output->scalar_type(), "rms_norm_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_in, true>;
        HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          output->DATA_PTR<scalar_t_out>(),
          invvar->DATA_PTR<accscalar_t>(),
          input->DATA_PTR<scalar_t_in>(),
          n1,n2,
          epsilon,
          gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL);
      )
}

// backward

template<typename T, typename U=float, typename V=T>
void HostRMSNormGradient(
    const V* dout,
    const U* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const V* gamma,
    double epsilon,
    T* grad_input,
    V* grad_gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL) {
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(U);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      // note (mkozuki): I can hard code part_grad_gamma's dtype as float given that
      // the `cuda_layer_norm_gradient` doesn't support double.
      const auto part_grad_dtype =
        (input->scalar_type() == at::ScalarType::Half || input->scalar_type() == at::ScalarType::BFloat16) ?
        at::ScalarType::Float :
        input->scalar_type();
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input->options().dtype(part_grad_dtype));
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                      dout,
                      input->DATA_PTR<T>(),
                      n1,n2,
                      invvar, // unused
                      invvar,
                      U(epsilon),
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_gamma.DATA_PTR<U>(), /* unused */
                      true);

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                      part_grad_gamma.DATA_PTR<U>(),
                      part_grad_gamma.DATA_PTR<U>(), /* unused */
                      part_size,
                      n1,n2,
                      grad_gamma,
                      grad_gamma, /* unused */
                      true);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    const dim3 threads1(32,4,1);
    int nshared =
            threads1.y > 1 ?
            threads1.y*threads1.x*sizeof(U) :
            0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input->DATA_PTR<T>(),
            n1,n2,
            invvar, /* unused */
            invvar,
            U(epsilon),
            gamma,
            grad_input,
            true);
}

void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma)
{
    using namespace at;
    // we can do away with `accscalar_t` as there're only three dtypes: fp32, fp16, bf16
    // DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      input->scalar_type(), gamma == NULL ? input->scalar_type() :  gamma->scalar_type(), "cuComputeGradInputRMS",
      using accscalar_t = at::acc_type<scalar_t_in, true>;
      HostRMSNormGradient(
        dout->DATA_PTR<scalar_t_out>(),
        invvar->DATA_PTR<accscalar_t>(),
        input,
        n1,n2,
            // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
            // if gamma Tensor is NULL on input.
        gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
        epsilon,
        grad_input->DATA_PTR<scalar_t_in>(),
        gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL);
    )
}


// 供cpp层调用函数，向外暴露的函数

void rms_norm_forward(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon,
    at::Tensor output,
    at::Tensor invvar) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  int n1,n2;
  check_args(input,normalized_shape,gamma,n1,n2);
//   at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
//   at::Tensor invvar = at::empty({n1}, input.options().dtype(stats_dtype));
  cuda_rms_norm(&output,&invvar,&input,n1,n2,
      normalized_shape,&gamma,epsilon);
//   return {output, invvar};
    return ;
}

void rms_norm_backward(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    double epsilon,
    at::Tensor grad_input,
    at::Tensor grad_gamma) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  int n1,n2;
  check_args(input,normalized_shape,gamma,n1,n2);
//   at::Tensor grad_input = at::empty_like(input);
//   at::Tensor grad_gamma = at::empty_like(gamma);
  cuda_rms_norm_gradient(&dout,&invvar,&input,n1,n2,
      normalized_shape,&gamma,epsilon,
      &grad_input,&grad_gamma);
//   return {grad_input, grad_gamma};
    return ;
}



}  // namespace ops
}  // namespace ext