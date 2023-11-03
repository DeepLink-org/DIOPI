/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/cuda/Loops.cuh>

namespace ext {
namespace ops {

void apply_rotary_cuda(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& cos, const at::Tensor& sin, at::Tensor out1, at::Tensor out2,
                       const bool conj) {
    auto iter = at::TensorIteratorConfig()
                    .add_output(out1)
                    .add_output(out2)
                    .add_input(x1)
                    .add_input(x2)
                    .add_input(cos)
                    .add_input(sin)
                    .check_all_same_dtype(false)
                    .promote_inputs_to_common_dtype(false)
                    .build();

    if (!conj) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x1.scalar_type(), "rotary_kernel", [&] {
            at::native::gpu_kernel_multiple_outputs(
                iter, [] GPU_LAMBDA(scalar_t x1, scalar_t x2, scalar_t cos, scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                    scalar_t out1 = static_cast<float>(x1) * static_cast<float>(cos) - static_cast<float>(x2) * static_cast<float>(sin);
                    scalar_t out2 = static_cast<float>(x1) * static_cast<float>(sin) + static_cast<float>(x2) * static_cast<float>(cos);
                    return {out1, out2};
                });
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x1.scalar_type(), "rotary_kernel", [&] {
            at::native::gpu_kernel_multiple_outputs(
                iter, [] GPU_LAMBDA(scalar_t x1, scalar_t x2, scalar_t cos, scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                    scalar_t out1 = static_cast<float>(x1) * static_cast<float>(cos) + static_cast<float>(x2) * static_cast<float>(sin);
                    scalar_t out2 = -static_cast<float>(x1) * static_cast<float>(sin) + static_cast<float>(x2) * static_cast<float>(cos);
                    return {out1, out2};
                });
        });
    }
}
}  // namespace ops
}  // namespace ext
