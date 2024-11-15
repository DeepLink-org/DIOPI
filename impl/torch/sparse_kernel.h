#pragma once

#include <ATen/ATen.h>

namespace sparse {
namespace ops {

at::Tensor row_balance_row_major_seq_reduce_kernel(at::Tensor& out, const at::Tensor& row_ptr, const at::Tensor& col_ind, const at::Tensor& value,
                                                   at::Tensor& input);

at::Tensor conv_forward_fetch_on_demand_cuda(
    at::Tensor in_feat, at::Tensor& out_feat, at::Tensor kernel, 
    at::Tensor neighbor_map, const int sum_nnz, 
    at::Tensor neighbor_address, at::Tensor q_neighbor_address,
    const int output_size, const int qsum_nnz, const bool transpose, 
    const bool allow_tf32, const bool allow_fp16);

// at::Tensor conv_forward_fetch_on_demand_no_fusion_cuda(
//     at::Tensor in_feat, at::Tensor kernel,
//     at::Tensor neighbor_map, at::Tensor neighbor_offset, 
//     const int sum_nnz, const int output_size, const bool transpose, 
//     const bool allow_tf32, const bool allow_fp16);

}  // namespace ops
}  // namespace sparse
