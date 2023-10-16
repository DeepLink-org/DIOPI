/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {

DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t argmaxY,
                                                    diopiTensorHandle_t argmaxX, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois,
                                                    int64_t alignedHeight, int64_t alignedWidth, int64_t samplingRatio, int64_t poolMode, float spatialScale,
                                                    bool aligned) {
int64_t roi_end_mode = 2;
  if (!aligned) {
    warning("The [aligned] attr in roi_align op is false");
    roi_end_mode = 0;
  }
//   int64_t aligned_height_64 = alignedHeight;
//   int64_t aligned_width_64 = alignedWidth;
//   int64_t sampling_ratio_64 = samplingRatio;
  AclOpRunner<2, 1>("ROIAlign", ctx).addInput(input)
      .addInput(rois)
      .setAttr<float>("spatial_scale", spatialScale)
      .setAttr<int64_t>("pooled_height", alignedHeight)
      .setAttr<int64_t>("pooled_width", alignedWidth)
      .setAttr<int64_t>("sample_num", samplingRatio)
      .setAttr<int64_t>("roi_end_mode", roi_end_mode)
      .addOutput(output)
      .run();
  return diopiSuccess;
}

// diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
//                                                   diopiConstTensorHandle_t rois, diopiConstTensorHandle_t argmaxY, diopiConstTensorHandle_t argmaxX,
//                                                   int64_t alignedHeight, int64_t alignedWidth, int64_t samplingRatio, int64_t poolMode, float spatialScale,
//                                                   bool aligned) {
//   int64_t roi_end_mode = 2;
//   if (!aligned) {
//     warning("The [aligned] attr in roi_align_grad op is false");
//     roi_end_mode = 0;
//   }
//   c10::SmallVector<int64_t, SIZE> xdiff_shape =
//       at_npu::native::array_to_small_vector(grad_input.sizes());

//   AclOpRunner<2, 1>("ROIAlignGrad", ctx).addInput(gradOutput)
//       .addInput(rois)
//       .addOutput(gradInput)
//       .setAttr("xdiff_shape", xdiff_shape)
//       .setAttr("pooled_width", alignedWidth)
//       .setAttr("pooled_height", alignedHeight)
//       .setAttr("spatial_scale", spatialScale)
//       .setAttr("sample_num", samplingRatio)
//       .setAttr("roi_end_mode", roi_end_mode)
//       .Run();
// }
}
}  // namespace ascend
}  // namespace impl
