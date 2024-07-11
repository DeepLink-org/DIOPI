
#pragma once

#include <ATen/ATen.h>

namespace sparse {
namespace ops {

at::Tensor row_balance_row_major_seq_reduce_kernel(at::Tensor& out,  at::Tensor& row_ptr,  at::Tensor& col_ind, at::Tensor& value,  at::Tensor& input);

}  // namespace ops
}  // namespace sparse
