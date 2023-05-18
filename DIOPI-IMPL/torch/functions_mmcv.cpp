/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "context.h"
#include "helper.hpp"
#include "mmcv_kernel.h"


extern "C" {

diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out,
                      diopiConstTensorHandle_t dets,
                      diopiConstTensorHandle_t scores, double iouThreshold,
                      int64_t offset) {
  auto atDets = impl::aten::buildATen(dets);
  auto atScores = impl::aten::buildATen(scores);
  auto atOut = mmcv::ops::NMSCUDAKernelLauncher(atDets, atScores, iouThreshold, offset);
  impl::aten::buildDiopiTensor(ctx, atOut, out);
}

diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output_, diopiTensorHandle_t argmax_y_,
                               diopiTensorHandle_t argmax_x_, diopiConstTensorHandle_t input_, diopiConstTensorHandle_t rois_,
                               int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio, int64_t pool_mode,
                               float spatial_scale, bool aligned) {
  auto input = impl::aten::buildATen(input_);
  auto rois = impl::aten::buildATen(rois_);
  auto output = impl::aten::buildATen(output_);
  auto argmax_y = impl::aten::buildATen(argmax_y_);
  auto argmax_x = impl::aten::buildATen(argmax_x_);
  mmcv::ops::ROIAlignForwardCUDAKernelLauncher(
      input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width,
      spatial_scale, sampling_ratio, pool_mode, aligned);
}

diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input_, diopiConstTensorHandle_t grad_output_,
                                       diopiConstTensorHandle_t rois_, diopiConstTensorHandle_t argmax_y_, diopiConstTensorHandle_t argmax_x_,
                                       int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio, int64_t pool_mode, float spatial_scale,
                                       bool aligned) {
  return diopiErrorOccurred;
  auto grad_output = impl::aten::buildATen(grad_output_);
  auto rois = impl::aten::buildATen(rois_);
  auto argmax_y = impl::aten::buildATen(argmax_y_);
  auto argmax_x = impl::aten::buildATen(argmax_x_);
  auto grad_input = impl::aten::buildATen(grad_input_);
  mmcv::ops::ROIAlignBackwardCUDAKernelLauncher(
      grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height,
      aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
}

diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx,
                                       diopiTensorHandle_t output_,
                                       diopiConstTensorHandle_t input_,
                                       diopiConstTensorHandle_t target_,
                                       diopiConstTensorHandle_t weight_,
                                       float gamma,
                                       float alpha) {
  auto output = impl::aten::buildATen(output_);
  auto input = impl::aten::buildATen(input_);
  auto target = impl::aten::buildATen(target_);
  auto weight = impl::aten::buildATen(weight_);
  mmcv::ops::SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output,
                                                       gamma, alpha);
}

diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx,
                                               diopiTensorHandle_t grad_input_,
                                               diopiConstTensorHandle_t input_,
                                               diopiConstTensorHandle_t target_,
                                               diopiConstTensorHandle_t weight_,
                                               float gamma,
                                               float alpha) {
  auto grad_input = impl::aten::buildATen(grad_input_);
  auto input = impl::aten::buildATen(input_);
  auto target = impl::aten::buildATen(target_);
  auto weight = impl::aten::buildATen(weight_);
  mmcv::ops::SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input,
                                                        gamma, alpha);
}

diopiError_t diopiHardVoxelizeMmcv(diopiContextHandle_t ctx,
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
  auto voxels = impl::aten::buildATen(voxels_);
  auto coors = impl::aten::buildATen(coors_);
  auto num_points_per_voxel = impl::aten::buildATen(num_points_per_voxel_);
  auto voxel_num = impl::aten::buildATen(voxel_num_);
  auto points = impl::aten::buildATen(points_);
  auto voxel_size = impl::aten::buildATen(voxel_size_);
  auto coors_range = impl::aten::buildATen(coors_range_);

  int64_t *voxel_num_data = voxel_num.data_ptr<int64_t>();
  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());

  if (deterministic) {
    *voxel_num_data = mmcv::ops::HardVoxelizeForwardCUDAKernelLauncher(
      points, voxels, coors, num_points_per_voxel, voxel_size_v, coors_range_v,
      max_points, max_voxels, NDim);
  } else {
    *voxel_num_data = mmcv::ops::NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
        points, voxels, coors, num_points_per_voxel, voxel_size_v,
        coors_range_v, max_points, max_voxels, NDim);
  }
}

/**
 * @brief Convert kitti points(N, >=3) to voxels(max_points == -1 or max_voxels
 * == -1).
 *  @param[in] ctx diopi context.
 * @sa definition of other parameters, refer to diopiHardVoxelize().
 */
diopiError_t diopiDynamicVoxelizeMmcv(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t coors_,
                                                diopiConstTensorHandle_t points_,
                                                diopiConstTensorHandle_t voxel_size_,
                                                diopiConstTensorHandle_t coors_range_,
                                                const int64_t NDim) {
  auto coors = impl::aten::buildATen(coors_);
  auto points = impl::aten::buildATen(points_);
  auto voxel_size = impl::aten::buildATen(voxel_size_);
  auto coors_range = impl::aten::buildATen(coors_range_);

  std::vector<float> voxel_size_v(
      voxel_size.data_ptr<float>(),
      voxel_size.data_ptr<float>() + voxel_size.numel());
  std::vector<float> coors_range_v(
      coors_range.data_ptr<float>(),
      coors_range.data_ptr<float>() + coors_range.numel());
  mmcv::ops::DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size_v, coors_range_v, NDim);
}

diopiError_t diopiModulatedDeformConvMmcv(
    diopiContextHandle_t ctx, diopiTensorHandle_t output,
    diopiTensorHandle_t columns, diopiTensorHandle_t ones,
    diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t offset,
    diopiConstTensorHandle_t mask, int64_t kernel_h, int64_t kernel_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t pad_h,
    const int64_t pad_w, const int64_t dilation_h, const int64_t dilation_w,
    const int64_t group, const int64_t deformable_group, const bool with_bias) {
    // stub here for nv
    return diopiErrorOccurred;
}

diopiError_t diopiModulatedDeformConvBackwardMmcv(
    diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
    diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
    diopiTensorHandle_t grad_offset, diopiTensorHandle_t grad_mask,
    diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t ones,
    diopiConstTensorHandle_t offset, diopiConstTensorHandle_t mask,
    diopiConstTensorHandle_t columns, diopiConstTensorHandle_t grad_output,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w, int64_t dilation_h, int64_t dilation_w,
    int64_t group, int64_t deformable_group, const bool with_bias) {
    // stub here for nv
    return diopiErrorOccurred;
}

}  // extern "C"