/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/library.h>

#include "cuda_helpers.h"
using namespace cuda::helper;

namespace mmcv {
namespace ops {

using namespace at;

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floorf((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      continue;
    }

    int c_y = floorf((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      continue;
    }

    int c_z = floorf((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if (coor_offset[0] == -1) continue;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          break;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

__global__ void nondeterministic_get_assign_pos(
    const int nthreads, const int32_t* coors_map, int32_t* pts_id,
    int32_t* coors_count, int32_t* reduce_count, int32_t* coors_order) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    if (coors_idx > -1) {
      int32_t coors_pts_pos = atomicAdd(&reduce_count[coors_idx], 1);
      pts_id[thread_idx] = coors_pts_pos;
      if (coors_pts_pos == 0) {
        coors_order[coors_idx] = atomicAdd(coors_count, 1);
      }
    }
  }
}

template <typename T>
__global__ void nondeterministic_assign_point_voxel(
    const int nthreads, const T* points, const int32_t* coors_map,
    const int32_t* pts_id, const int32_t* coors_in, const int32_t* reduce_count,
    const int32_t* coors_order, T* voxels, int32_t* coors, int32_t* pts_count,
    const int max_voxels, const int max_points, const int num_features,
    const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    int coors_pts_pos = pts_id[thread_idx];
    if (coors_idx > -1 && coors_pts_pos < max_points) {
      int coors_pos = coors_order[coors_idx];
      if (coors_pos < max_voxels) {
        auto voxels_offset =
            voxels + (coors_pos * max_points + coors_pts_pos) * num_features;
        auto points_offset = points + thread_idx * num_features;
        for (int k = 0; k < num_features; k++) {
          voxels_offset[k] = points_offset[k];
        }
        if (coors_pts_pos == 0) {
          pts_count[coors_pos] = min(reduce_count[coors_idx], max_points);
          auto coors_offset = coors + coors_pos * NDim;
          auto coors_in_offset = coors_in + coors_idx * NDim;
          for (int k = 0; k < NDim; k++) {
            coors_offset[k] = coors_in_offset[k];
          }
        }
      }
    }
  }
}

int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor& points, at::Tensor& voxels, at::Tensor& coors,
    at::Tensor& num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int><<<grid, block, 0, stream>>>(
            points.contiguous().data_ptr<scalar_t>(),
            temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto point_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));

  dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);

  AT_DISPATCH_ALL_TYPES(
      temp_coors.scalar_type(), "determin_duplicate", ([&] {
        point_to_voxelidx_kernel<int><<<map_grid, map_block, 0, stream>>>(
            temp_coors.contiguous().data_ptr<int>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            point_to_pointidx.contiguous().data_ptr<int>(), max_points,
            max_voxels, num_points, NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 3. determine voxel num and voxel's coor index
  // make the logic in the CUDA device could accelerate about 10 times
  auto coor_to_voxelidx = -at::ones(
      {
          num_points,
      },
      points.options().dtype(at::kInt));
  auto voxel_num = at::zeros(
      {
          1,
      },
      points.options().dtype(at::kInt));  // must be zero from the beginning

  AT_DISPATCH_ALL_TYPES(temp_coors.scalar_type(), "determin_duplicate", ([&] {
                          determin_voxel_num<int><<<1, 1, 0, stream>>>(
                              num_points_per_voxel.contiguous().data_ptr<int>(),
                              point_to_voxelidx.contiguous().data_ptr<int>(),
                              point_to_pointidx.contiguous().data_ptr<int>(),
                              coor_to_voxelidx.contiguous().data_ptr<int>(),
                              voxel_num.contiguous().data_ptr<int>(),
                              max_points, max_voxels, num_points);
                        }));

  AT_CUDA_CHECK(cudaGetLastError());

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_point_to_voxel<float, int><<<cp_grid, cp_block, 0, stream>>>(
            pts_output_size, points.contiguous().data_ptr<float>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            coor_to_voxelidx.contiguous().data_ptr<int>(),
            voxels.contiguous().data_ptr<float>(), max_points, num_features,
            num_points, NDim);
      }));
  //   cudaDeviceSynchronize();
  //   AT_CUDA_CHECK(cudaGetLastError());

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(
      std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_voxel_coors<float, int>
            <<<coors_cp_grid, coors_cp_block, 0, stream>>>(
                coors_output_size, temp_coors.contiguous().data_ptr<int>(),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                coor_to_voxelidx.contiguous().data_ptr<int>(),
                coors.contiguous().data_ptr<int>(), num_points, NDim);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];

  return voxel_num_int;
}

int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor& points, at::Tensor& voxels, at::Tensor& coors,
    at::Tensor& num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  if (num_points == 0) return 0;

  dim3 blocks(
      std::min(at::cuda::ATenCeilDiv(num_points, THREADS_PER_BLOCK), 4096));
  dim3 threads(THREADS_PER_BLOCK);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
            points.contiguous().data_ptr<scalar_t>(),
            temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
      }));

  at::Tensor coors_map;
  at::Tensor reduce_count;

  auto coors_clean = temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);

  std::tie(temp_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, false);

  if (temp_coors[0][0].lt(0).item<bool>()) {
    // the first element of temp_coors is (-1,-1,-1) and should be removed
    temp_coors = temp_coors.slice(0, 1);
    coors_map = coors_map - 1;
  }

  int num_coors = temp_coors.size(0);
  temp_coors = temp_coors.to(at::kInt);
  coors_map = coors_map.to(at::kInt);

  at::Tensor coors_count = at::zeros({1}, coors_map.options());
  at::Tensor coors_order = at::empty({num_coors}, coors_map.options());
  at::Tensor pts_id = at::zeros({num_points}, coors_map.options());
  reduce_count = at::zeros({num_coors}, coors_map.options());

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "get_assign_pos", ([&] {
        nondeterministic_get_assign_pos<<<blocks, threads, 0, stream>>>(
            num_points, coors_map.contiguous().data_ptr<int32_t>(),
            pts_id.contiguous().data_ptr<int32_t>(),
            coors_count.contiguous().data_ptr<int32_t>(),
            reduce_count.contiguous().data_ptr<int32_t>(),
            coors_order.contiguous().data_ptr<int32_t>());
      }));

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        nondeterministic_assign_point_voxel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                num_points, points.contiguous().data_ptr<scalar_t>(),
                coors_map.contiguous().data_ptr<int32_t>(),
                pts_id.contiguous().data_ptr<int32_t>(),
                temp_coors.contiguous().data_ptr<int32_t>(),
                reduce_count.contiguous().data_ptr<int32_t>(),
                coors_order.contiguous().data_ptr<int32_t>(),
                voxels.contiguous().data_ptr<scalar_t>(),
                coors.contiguous().data_ptr<int32_t>(),
                num_points_per_voxel.contiguous().data_ptr<int32_t>(),
                max_voxels, max_points, num_features, NDim);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return max_voxels < num_coors ? max_voxels : num_coors;
}

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor& points, at::Tensor& coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device

  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = at::cuda::ATenCeilDiv(num_points, THREADS_PER_BLOCK);
  dim3 blocks(col_blocks);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
  });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ops
}  // namespace mmcv
