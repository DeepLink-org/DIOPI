/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <memory>
#include <cmath>

#include "../mlu_helper.hpp"
#include "../diopi_helper.hpp"

namespace impl {

namespace camb {

void KernelDynamicVoxelize(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, void *coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int32_t grid_x, const int32_t grid_y,
    const int32_t grid_z, const int32_t num_points, const int32_t num_features);

void KernelPoint2Voxel(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                       cnrtQueue_t queue, void *coors, void *point_to_pointidx,
                       void *point_to_voxelidx, const int32_t num_points,
                       const int32_t max_points);

void KernelCalcPointsPerVoxel(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                              cnrtQueue_t queue, void *point_to_pointidx,
                              void *point_to_voxelidx, void *coor_to_voxelidx,
                              void *num_points_per_voxel, void *voxel_num,
                              const int32_t max_voxels,
                              const int32_t num_points);

void KernelAssignVoxelsCoors(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                             cnrtQueue_t queue, const void *points,
                             void *temp_coors, void *point_to_voxelidx,
                             void *coor_to_voxelidx, void *voxels, void *coors,
                             const int32_t max_points, const int32_t num_points,
                             const int32_t num_features);
}  // namespace camb

}  // namespace impl

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// policy function
static void policyFuncDefault(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                              const int num_points) {
  k_dim->x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = MIN((num_points + k_dim->x - 1) / k_dim->x,
                 impl::camb::getDeviceAttr(cnrtAttrClusterCount));
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

// policy function
static void policyFuncCalcPointsPerVoxel(cnrtDim3_t *k_dim,
                                         cnrtFunctionType_t *k_type,
                                         const int num_points) {
  k_dim->x = 1;
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_BLOCK;
}

extern "C" diopiError_t diopiHardVoxelizeMmcv(diopiContextHandle_t ctx,
                                             diopiTensorHandle_t voxels_,
                                             diopiTensorHandle_t coors_,
                                             diopiTensorHandle_t num_points_per_voxel_,
                                             diopiTensorHandle_t voxel_num_,
                                             diopiConstTensorHandle_t points_,
                                             diopiConstTensorHandle_t voxel_size_,
                                             diopiConstTensorHandle_t coors_range_,
                                             const int64_t max_points,
                                             const int64_t max_voxels,
                                             const int64_t NDim,
                                             const bool deterministic) {
  auto voxels = impl::camb::DiopiTensor(voxels_);
  auto coors = impl::camb::DiopiTensor(coors_);
  auto num_points_per_voxel = impl::camb::DiopiTensor(num_points_per_voxel_);
  auto voxel_num = impl::camb::DiopiTensor(voxel_num_);
  auto points = impl::camb::DiopiTensor(points_);
  auto voxel_size = impl::camb::DiopiTensor(voxel_size_);
  auto coors_range = impl::camb::DiopiTensor(coors_range_);

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  int64_t *voxel_num_data = reinterpret_cast<int64_t*>(voxel_num.data());
  std::vector<float> voxel_size_v(
      reinterpret_cast<float*>(voxel_size.data()),
      reinterpret_cast<float*>(voxel_size.data()) + voxel_size.numel());
  std::vector<float> coors_range_v(
      reinterpret_cast<float*>(coors_range.data()),
      reinterpret_cast<float*>(coors_range.data()) + coors_range.numel());

  // check zero element
  if (max_points == 0 || max_voxels == 0) {
   *voxel_num_data = 0;
    return diopiSuccess;
  }

  // get compute queue
  auto queue = impl::camb::getStream(ctx);

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncDefault(&k_dim, &k_type, num_points);

  // 1. link point to corresponding voxel coors
  const float voxel_x = voxel_size_v[0];
  const float voxel_y = voxel_size_v[1];
  const float voxel_z = voxel_size_v[2];
  const float coors_x_min = coors_range_v[0];
  const float coors_y_min = coors_range_v[1];
  const float coors_z_min = coors_range_v[2];
  const float coors_x_max = coors_range_v[3];
  const float coors_y_max = coors_range_v[4];
  const float coors_z_max = coors_range_v[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  diopiScalar_t scalar = {diopi_dtype_int32, 0};
  auto temp_coors = impl::camb::requiresTensor(ctx, {NDim, num_points}, diopi_dtype_int32);
  DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(temp_coors), &scalar));

  impl::camb::KernelDynamicVoxelize(k_dim, k_type, queue, points.data(), temp_coors.data(),
                        voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min,
                        coors_z_min, coors_x_max, coors_y_max, coors_z_max,
                        grid_x, grid_y, grid_z, num_points, num_features);

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  auto point_to_pointidx =  impl::camb::requiresTensor(ctx, {num_points}, diopi_dtype_int32);
  DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(point_to_pointidx), &scalar));

  auto point_to_voxelidx = impl::camb::requiresTensor(ctx, {num_points}, diopi_dtype_int32);
  DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(point_to_voxelidx), &scalar));

  impl::camb::KernelPoint2Voxel(k_dim, k_type, queue, temp_coors.data(), point_to_pointidx.data(),
                    point_to_voxelidx.data(), num_points, max_points);

  // calculate task dimension
  cnrtDim3_t k_dim_calc_points_per_voxel;
  cnrtFunctionType_t k_type_calc_points_per_voxel;
  policyFuncCalcPointsPerVoxel(&k_dim_calc_points_per_voxel,
                               &k_type_calc_points_per_voxel, num_points);

  // 3. determine voxel num and voxel's coor index
  auto coor_to_voxelidx =impl::camb::requiresTensor(ctx, {num_points}, diopi_dtype_int32);
  DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(coor_to_voxelidx), &scalar));
  auto voxel_num_tmp = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int32);
  DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(voxel_num_tmp), &scalar));

  impl::camb::KernelCalcPointsPerVoxel(
      k_dim_calc_points_per_voxel, k_type_calc_points_per_voxel, queue,
      point_to_pointidx.data(), point_to_voxelidx.data(), coor_to_voxelidx.data(),
      num_points_per_voxel.data(), voxel_num_tmp.data(), max_voxels, num_points);

  // 4. copy point features and coors of each voxels to voxels
  impl::camb::KernelAssignVoxelsCoors(k_dim, k_type, queue, points.data(), temp_coors.data(),
                          point_to_voxelidx.data(), coor_to_voxelidx.data(),
                          voxels.data(), coors.data(), max_points, num_points,
                          num_features);

  int bytes = sizeof(int) * voxel_num_tmp.numel();
  std::unique_ptr<char> voxel_num_cpu(new char[bytes]);
  cnrtMemcpyAsync(voxel_num_cpu.get(), voxel_num_tmp.data(), bytes, impl::camb::getStream(ctx), cnrtMemcpyDevToHost);
  impl::camb::syncStreamInCtx(ctx);
  int voxel_num_int = reinterpret_cast<int*>(voxel_num_cpu.get())[0];
  *voxel_num_data = voxel_num_int;
  return diopiSuccess;

}
