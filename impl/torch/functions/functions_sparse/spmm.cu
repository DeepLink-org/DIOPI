#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "utils.h"

namespace sparse{
namespace ops {


template <typename Index, typename DType>
__global__ void
csrspmm_seqreduce_rowbalance_kernel(const Index nr, const Index feature_size,
                                    const Index rowPtr[], const Index colIdx[],
                                    const DType values[], const DType dnInput[],
                                    DType dnOutput[]) {
	Index row_tile = blockDim.y; // 8
	Index subwarp_id = threadIdx.y;
	Index stride = row_tile * gridDim.x; // 8 * (m/8)
	Index row = blockIdx.x * row_tile + subwarp_id;
	Index v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
	dnInput += v_id;
	dnOutput += v_id;
	DType val;
	// DType res = init(REDUCE::Op);
	Index col;
	for (; row < nr; row += stride) {
		DType res = 0;
		Index E_k_idx = -1;
		Index start = __ldg(rowPtr + row);
		Index end = __ldg(rowPtr + row + 1);

		for (Index p = start; p < end; p++) {
			col = __ldg(colIdx + p);
			val = __guard_load_default_one<DType>(values, p);
			res += val * __ldg(dnInput + col * feature_size);
		}

		dnOutput[row * feature_size] = res;
	}
}

at::Tensor row_balance_row_major_seq_reduce_kernel(at::Tensor& out,  at::Tensor& row_ptr,  at::Tensor& col_ind,
                                                     at::Tensor& value,  at::Tensor& input){
	//   assertTensor(row_ptr, at::kInt32);
	//   assertTensor(col_ind, at::kInt32);
	//   assertTensor(input, at::kFloat32);
	//   assertTensor(value, at::kFloat32);
	input = input.contiguous();
	//   int v = row_ptr.size(0) - 1;
	// 	 int Ndim_worker = input.size(1);
	//   int f = Ndim_worker;
	//   int e = col_ind.size(0);

	int Mdim_worker = row_ptr.size(0) - 1;
	// int v = Mdim_worker;
	int Ndim_worker = input.size(1);
	// int f = Ndim_worker;
	// int e = col_ind.size(0);
	int RefThreadPerBlock = 256;
	int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
	int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
	int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
	int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

	dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
	dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

	// auto out = at::empty({v, f}, options);
	csrspmm_seqreduce_rowbalance_kernel<int, float>
		<<<gridDim, blockDim>>>(
			Mdim_worker, Ndim_worker, row_ptr.data_ptr<int>(),
			col_ind.data_ptr<int>(), value.data_ptr<float>(),
			input.data_ptr<float>(), out.data_ptr<float>());
	return out;
}

}
}