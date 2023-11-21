/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <iostream>

#include "../context.h"
#include "../helper.hpp"
#include "../mmcv_kernel.h"

namespace impl {
namespace cuda {

diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores,
                          double iouThreshold, int64_t offset) {
    impl::aten::setCurCtx(ctx);
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOut = mmcv::ops::NMSCUDAKernelLauncher(atDets, atScores, iouThreshold, offset);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNmsRotatedMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores,
                                 diopiConstTensorHandle_t order, diopiConstTensorHandle_t detsSorted, diopiConstTensorHandle_t labels, float iouThreshold,
                                 bool multiLabel) {
    impl::aten::setCurCtx(ctx);
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOrder = impl::aten::buildATen(order);
    auto atDetsSorted = impl::aten::buildATen(detsSorted);
    auto atOut = mmcv::ops::NMSRotatedCUDAKernelLauncher(atDets, atScores, atOrder, atDetsSorted, iouThreshold, multiLabel);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output_, diopiTensorHandle_t argmax_y_, diopiTensorHandle_t argmax_x_,
                               diopiConstTensorHandle_t input_, diopiConstTensorHandle_t rois_, int64_t aligned_height, int64_t aligned_width,
                               int64_t sampling_ratio, int64_t pool_mode, float spatial_scale, bool aligned) {
    impl::aten::setCurCtx(ctx);
    auto input = impl::aten::buildATen(input_);
    auto rois = impl::aten::buildATen(rois_);
    auto output = impl::aten::buildATen(output_);
    auto argmax_y = impl::aten::buildATen(argmax_y_);
    auto argmax_x = impl::aten::buildATen(argmax_x_);
    mmcv::ops::ROIAlignForwardCUDAKernelLauncher(
        input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
    return diopiSuccess;
}

diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input_, diopiConstTensorHandle_t grad_output_,
                                       diopiConstTensorHandle_t rois_, diopiConstTensorHandle_t argmax_y_, diopiConstTensorHandle_t argmax_x_,
                                       int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio, int64_t pool_mode, float spatial_scale,
                                       bool aligned) {
    impl::aten::setCurCtx(ctx);
    auto grad_output = impl::aten::buildATen(grad_output_);
    auto rois = impl::aten::buildATen(rois_);
    auto argmax_y = impl::aten::buildATen(argmax_y_);
    auto argmax_x = impl::aten::buildATen(argmax_x_);
    auto grad_input = impl::aten::buildATen(grad_input_);
    mmcv::ops::ROIAlignBackwardCUDAKernelLauncher(
        grad_output, rois, argmax_y, argmax_x, grad_input, aligned_height, aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned);
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output_, diopiConstTensorHandle_t input_, diopiConstTensorHandle_t target_,
                                       diopiConstTensorHandle_t weight_, float gamma, float alpha) {
    impl::aten::setCurCtx(ctx);
    auto output = impl::aten::buildATen(output_);
    auto input = impl::aten::buildATen(input_);
    auto target = impl::aten::buildATen(target_);
    auto weight = impl::aten::buildATen(weight_);
    mmcv::ops::SigmoidFocalLossForwardCUDAKernelLauncher(input, target, weight, output, gamma, alpha);
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input_, diopiConstTensorHandle_t input_,
                                               diopiConstTensorHandle_t target_, diopiConstTensorHandle_t weight_, float gamma, float alpha) {
    impl::aten::setCurCtx(ctx);
    auto grad_input = impl::aten::buildATen(grad_input_);
    auto input = impl::aten::buildATen(input_);
    auto target = impl::aten::buildATen(target_);
    auto weight = impl::aten::buildATen(weight_);
    mmcv::ops::SigmoidFocalLossBackwardCUDAKernelLauncher(input, target, weight, grad_input, gamma, alpha);
    return diopiSuccess;
}

diopiError_t diopiHardVoxelizeMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t voxels_, diopiTensorHandle_t coors_, diopiTensorHandle_t num_points_per_voxel_,
                                   diopiTensorHandle_t voxel_num_, diopiConstTensorHandle_t points_, diopiConstTensorHandle_t voxel_size_,
                                   diopiConstTensorHandle_t coors_range_, const int64_t max_points, const int64_t max_voxels, const int64_t NDim,
                                   const bool deterministic) {
    impl::aten::setCurCtx(ctx);
    auto voxels = impl::aten::buildATen(voxels_);
    auto coors = impl::aten::buildATen(coors_);
    auto num_points_per_voxel = impl::aten::buildATen(num_points_per_voxel_);
    auto voxel_num = impl::aten::buildATen(voxel_num_);
    auto points = impl::aten::buildATen(points_);
    auto voxel_size = impl::aten::buildATen(voxel_size_);
    auto coors_range = impl::aten::buildATen(coors_range_);

    int64_t *voxel_num_data = voxel_num.data_ptr<int64_t>();
    std::vector<float> voxel_size_v(voxel_size.data_ptr<float>(), voxel_size.data_ptr<float>() + voxel_size.numel());
    std::vector<float> coors_range_v(coors_range.data_ptr<float>(), coors_range.data_ptr<float>() + coors_range.numel());

    if (deterministic) {
        *voxel_num_data = mmcv::ops::HardVoxelizeForwardCUDAKernelLauncher(
            points, voxels, coors, num_points_per_voxel, voxel_size_v, coors_range_v, max_points, max_voxels, NDim);
    } else {
        *voxel_num_data = mmcv::ops::NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
            points, voxels, coors, num_points_per_voxel, voxel_size_v, coors_range_v, max_points, max_voxels, NDim);
    }
    return diopiSuccess;
}

/**
 * @brief Convert kitti points(N, >=3) to voxels(max_points == -1 or max_voxels
 * == -1).
 *  @param[in] ctx diopi context.
 * @sa definition of other parameters, refer to diopiHardVoxelize().
 */
diopiError_t diopiDynamicVoxelizeMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t coors_, diopiConstTensorHandle_t points_,
                                      diopiConstTensorHandle_t voxel_size_, diopiConstTensorHandle_t coors_range_, const int64_t NDim) {
    impl::aten::setCurCtx(ctx);
    auto coors = impl::aten::buildATen(coors_);
    auto points = impl::aten::buildATen(points_);
    auto voxel_size = impl::aten::buildATen(voxel_size_);
    auto coors_range = impl::aten::buildATen(coors_range_);

    std::vector<float> voxel_size_v(voxel_size.data_ptr<float>(), voxel_size.data_ptr<float>() + voxel_size.numel());
    std::vector<float> coors_range_v(coors_range.data_ptr<float>(), coors_range.data_ptr<float>() + coors_range.numel());
    mmcv::ops::DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size_v, coors_range_v, NDim);
    return diopiSuccess;
}

diopiError_t diopiModulatedDeformConvMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output_, diopiTensorHandle_t columns_, diopiTensorHandle_t ones_,
                                          diopiConstTensorHandle_t input_, diopiConstTensorHandle_t weight_, diopiConstTensorHandle_t bias_,
                                          diopiConstTensorHandle_t offset_, diopiConstTensorHandle_t mask_, int64_t kernel_h, int64_t kernel_w,
                                          const int64_t stride_h, const int64_t stride_w, const int64_t pad_h, const int64_t pad_w, const int64_t dilation_h,
                                          const int64_t dilation_w, const int64_t group, const int64_t deformable_group, const bool with_bias) {
    impl::aten::setCurCtx(ctx);
    auto output = impl::aten::buildATen(output_);
    auto columns = impl::aten::buildATen(columns_);
    auto ones = impl::aten::buildATen(ones_);
    auto input = impl::aten::buildATen(input_);
    auto weight = impl::aten::buildATen(weight_);
    auto bias = impl::aten::buildATen(bias_);
    auto offset = impl::aten::buildATen(offset_);
    auto mask = impl::aten::buildATen(mask_);

    at::DeviceGuard guard(input.device());

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel * group) AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 || ones.size(0) * ones.size(1) < height_out * width_out) {
        // Resize plane and fill with ones...
        ones = at::ones({height_out, width_out}, input.options());
    }

    // resize output
    output = output.view({batch, channels_out, height_out, width_out}).zero_();
    // resize temporary columns
    columns = at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());

    output = output.view({output.size(0), group, output.size(1) / group, output.size(2), output.size(3)});

    for (int b = 0; b < batch; b++) {
        mmcv::ops::modulated_deformable_im2col_cuda(input[b],
                                                    offset[b],
                                                    mask[b],
                                                    1,
                                                    channels,
                                                    height,
                                                    width,
                                                    height_out,
                                                    width_out,
                                                    kernel_h,
                                                    kernel_w,
                                                    pad_h,
                                                    pad_w,
                                                    stride_h,
                                                    stride_w,
                                                    dilation_h,
                                                    dilation_w,
                                                    deformable_group,
                                                    columns);

        // divide into group
        weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});
        columns = columns.view({group, columns.size(0) / group, columns.size(1)});

        for (int g = 0; g < group; g++) {
            output[b][g] = output[b][g].flatten(1).addmm_(weight[g].flatten(1), columns[g]).view_as(output[b][g]);
        }

        weight = weight.view({weight.size(0) * weight.size(1), weight.size(2), weight.size(3), weight.size(4)});
        columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    }

    output = output.view({output.size(0), output.size(1) * output.size(2), output.size(3), output.size(4)});

    if (with_bias) {
        output += bias.view({1, bias.size(0), 1, 1});
    }
    return diopiSuccess;
}

diopiError_t diopiModulatedDeformConvBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input_, diopiTensorHandle_t grad_weight_,
                                                  diopiTensorHandle_t grad_bias_, diopiTensorHandle_t grad_offset_, diopiTensorHandle_t grad_mask_,
                                                  diopiConstTensorHandle_t input_, diopiConstTensorHandle_t weight_, diopiConstTensorHandle_t bias_,
                                                  diopiConstTensorHandle_t ones_, diopiConstTensorHandle_t offset_, diopiConstTensorHandle_t mask_,
                                                  diopiConstTensorHandle_t columns_, diopiConstTensorHandle_t grad_output_, int64_t kernel_h, int64_t kernel_w,
                                                  int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dilation_h, int64_t dilation_w,
                                                  int64_t group, int64_t deformable_group, const bool with_bias) {
    impl::aten::setCurCtx(ctx);
    auto grad_input = impl::aten::buildATen(grad_input_);
    auto grad_weight = impl::aten::buildATen(grad_weight_);
    auto grad_bias = impl::aten::buildATen(grad_bias_);
    auto grad_offset = impl::aten::buildATen(grad_offset_);
    auto grad_mask = impl::aten::buildATen(grad_mask_);
    auto input = impl::aten::buildATen(input_);
    auto weight = impl::aten::buildATen(weight_);
    auto bias = impl::aten::buildATen(bias_);
    auto ones = impl::aten::buildATen(ones_);
    auto offset = impl::aten::buildATen(offset_);
    auto mask = impl::aten::buildATen(mask_);
    auto columns = impl::aten::buildATen(columns_);
    auto grad_output = impl::aten::buildATen(grad_output_);

    at::DeviceGuard guard(input.device());

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);
    if (channels != channels_kernel * group) AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 || ones.size(0) * ones.size(1) < height_out * width_out) {
        // Resize plane and fill with ones...
        ones = at::ones({height_out, width_out}, input.options());
    }

    grad_input = grad_input.view({batch, channels, height, width});
    columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out}, input.options());

    grad_output = grad_output.view({grad_output.size(0), group, grad_output.size(1) / group, grad_output.size(2), grad_output.size(3)});

    for (int b = 0; b < batch; b++) {
        // divide int group
        columns = columns.view({group, columns.size(0) / group, columns.size(1)});
        weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});

        for (int g = 0; g < group; g++) {
            columns[g].addmm_(weight[g].flatten(1).transpose(0, 1), grad_output[b][g].flatten(1), 0.0f, 1.0f);
        }

        columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
        weight = weight.view({weight.size(0) * weight.size(1), weight.size(2), weight.size(3), weight.size(4)});

        // gradient w.r.t. input coordinate data
        mmcv::ops::modulated_deformable_col2im_coord_cuda(columns,
                                                          input[b],
                                                          offset[b],
                                                          mask[b],
                                                          1,
                                                          channels,
                                                          height,
                                                          width,
                                                          height_out,
                                                          width_out,
                                                          kernel_h,
                                                          kernel_w,
                                                          pad_h,
                                                          pad_w,
                                                          stride_h,
                                                          stride_w,
                                                          dilation_h,
                                                          dilation_w,
                                                          deformable_group,
                                                          grad_offset[b],
                                                          grad_mask[b]);
        // gradient w.r.t. input data
        mmcv::ops::modulated_deformable_col2im_cuda(columns,
                                                    offset[b],
                                                    mask[b],
                                                    1,
                                                    channels,
                                                    height,
                                                    width,
                                                    height_out,
                                                    width_out,
                                                    kernel_h,
                                                    kernel_w,
                                                    pad_h,
                                                    pad_w,
                                                    stride_h,
                                                    stride_w,
                                                    dilation_h,
                                                    dilation_w,
                                                    deformable_group,
                                                    grad_input[b]);

        // gradient w.r.t. weight, dWeight should accumulate across the batch and
        // group
        mmcv::ops::modulated_deformable_im2col_cuda(input[b],
                                                    offset[b],
                                                    mask[b],
                                                    1,
                                                    channels,
                                                    height,
                                                    width,
                                                    height_out,
                                                    width_out,
                                                    kernel_h,
                                                    kernel_w,
                                                    pad_h,
                                                    pad_w,
                                                    stride_h,
                                                    stride_w,
                                                    dilation_h,
                                                    dilation_w,
                                                    deformable_group,
                                                    columns);

        columns = columns.view({group, columns.size(0) / group, columns.size(1)});
        grad_weight = grad_weight.view({group, grad_weight.size(0) / group, grad_weight.size(1), grad_weight.size(2), grad_weight.size(3)});
        if (with_bias) grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

        for (int g = 0; g < group; g++) {
            grad_weight[g] = grad_weight[g].flatten(1).addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1)).view_as(grad_weight[g]);
            if (with_bias) {
                grad_bias[g] = grad_bias[g].view({-1, 1}).addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1})).view(-1);
            }
        }

        columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
        grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1), grad_weight.size(2), grad_weight.size(3), grad_weight.size(4)});
        if (with_bias) grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
    }
    grad_output = grad_output.view({grad_output.size(0) * grad_output.size(1), grad_output.size(2), grad_output.size(3), grad_output.size(4)});
    return diopiSuccess;
}

}  // namespace cuda
}  // namespace impl
