/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include "../common/acloprunner.hpp"

using namespace impl::ascend;

extern "C" {

DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t argmaxY, diopiTensorHandle_t argmaxX,
                                         diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, int64_t alignedHeight, int64_t alignedWidth,
                                         int64_t samplingRatio, int64_t poolMode, float spatialScale, bool aligned) {
    int64_t roiEndMode = 2;
    if (!aligned) {
        warning("The [aligned] attr in roi_align op is false");
        roiEndMode = 0;
    }

    diopiDtype_t inputDtype;
    diopiDtype_t roisDtype;
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(rois, &roisDtype);
    diopiTensorHandle_t outCopy;
    bool isHalf = inputDtype == diopi_dtype_float16 || roisDtype == diopi_dtype_float16;
    if (isHalf) {
        makeTensorLike(ctx, &outCopy, output, diopi_dtype_float32);
    } else {
        outCopy = output;
    }

    AclOpRunner<2, 1>("ROIAlign", ctx)
        .addInput(input, diopi_dtype_float32)
        .addInput(rois, diopi_dtype_float32)
        .setAttr<float>("spatial_scale", spatialScale)
        .setAttr<int64_t>("pooled_height", alignedHeight)
        .setAttr<int64_t>("pooled_width", alignedWidth)
        .setAttr<int64_t>("sample_num", samplingRatio)
        .setAttr<int64_t>("roi_end_mode", roiEndMode)
        .addOutput(outCopy)
        .run();

    if (isHalf) diopiCastDtype(ctx, output, outCopy);
    return diopiSuccess;
}

diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                       diopiConstTensorHandle_t rois, diopiConstTensorHandle_t argmaxY, diopiConstTensorHandle_t argmaxX, int64_t alignedHeight,
                                       int64_t alignedWidth, int64_t samplingRatio, int64_t poolMode, float spatialScale, bool aligned) {
    int64_t roiEndMode = 2;
    if (!aligned) {
        warning("The [aligned] attr in roi_align_grad op is false");
        roiEndMode = 0;
    }
    diopiSize_t xdiffShape;
    diopiGetTensorShape(gradInput, &xdiffShape);

    AclOpRunner<2, 1>("ROIAlignGrad", ctx)
        .addInput(gradOutput)
        .addInput(rois)
        .addOutput(gradInput)
        .setAttr("xdiff_shape", xdiffShape)
        .setAttr<int64_t>("pooled_width", alignedWidth)
        .setAttr<int64_t>("pooled_height", alignedHeight)
        .setAttr<float>("spatial_scale", spatialScale)
        .setAttr<int64_t>("sample_num", samplingRatio)
        .setAttr<int64_t>("roi_end_mode", roiEndMode)
        .run();

    return diopiSuccess;
}
}
