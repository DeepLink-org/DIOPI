/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_TORCH_MMCV_KERNEL_H_
#define IMPL_TORCH_MMCV_KERNEL_H_

#include <ATen/ATen.h>

#include <vector>

namespace mmcv {
namespace ops {

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold, int offset);

void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output, Tensor argmax_y, Tensor argmax_x, int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio, int pool_mode, bool aligned);

void ROIAlignBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois, Tensor argmax_y, Tensor argmax_x, Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale, int sampling_ratio, int pool_mode, bool aligned);

void SigmoidFocalLossForwardCUDAKernelLauncher(Tensor input, Tensor target, Tensor weight, Tensor output, const float gamma, const float alpha);

void SigmoidFocalLossBackwardCUDAKernelLauncher(Tensor input, Tensor target, Tensor weight, Tensor grad_input, const float gamma, const float alpha);

void SoftmaxFocalLossForwardCUDAKernelLauncher(Tensor softmax, Tensor target, Tensor weight, Tensor output, const float gamma, const float alpha);

void SoftmaxFocalLossBackwardCUDAKernelLauncher(Tensor softmax, Tensor target, Tensor weight, Tensor buff, Tensor grad_input, const float gamma,
                                                const float alpha);

int HardVoxelizeForwardCUDAKernelLauncher(const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors, at::Tensor &num_points_per_voxel,
                                          const std::vector<float> voxel_size, const std::vector<float> coors_range, const int max_points, const int max_voxels,
                                          const int NDim = 3);

int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors, at::Tensor &num_points_per_voxel,
                                                          const std::vector<float> voxel_size, const std::vector<float> coors_range, const int max_points,
                                                          const int max_voxels, const int NDim = 3);

void DynamicVoxelizeForwardCUDAKernelLauncher(const at::Tensor &points, at::Tensor &coors, const std::vector<float> voxel_size,
                                              const std::vector<float> coors_range, const int NDim = 3);

void modulated_deformable_im2col_cuda(const Tensor data_im, const Tensor data_offset, const Tensor data_mask, const int batch_size, const int channels,
                                      const int height_im, const int width_im, const int height_col, const int width_col, const int kernel_h,
                                      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
                                      const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_cuda(const Tensor data_col, const Tensor data_offset, const Tensor data_mask, const int batch_size, const int channels,
                                      const int height_im, const int width_im, const int height_col, const int width_col, const int kernel_h,
                                      const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h,
                                      const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(const Tensor data_col, const Tensor data_im, const Tensor data_offset, const Tensor data_mask, const int batch_size,
                                            const int channels, const int height_im, const int width_im, const int height_col, const int width_col,
                                            const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                            const int dilation_h, const int dilation_w, const int deformable_group, Tensor grad_offset, Tensor grad_mask);

}  // namespace ops
}  // namespace mmcv
#endif  // IMPL_TORCH_MMCV_KERNEL_H_
