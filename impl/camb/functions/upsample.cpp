/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t upsampleInternal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, bool alignCorners, bool alignCenter,
                              cnnlInterpMode_t interpMode) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    auto dim = inputTensor.dim();
    cnnlTensorLayout_t layout;
    if (3 == dim) {
        DIOPI_CHECK(inputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast1d");
        DIOPI_CHECK(outputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "outputTensor's memory format should be channelsLast1d");
        layout = CNNL_LAYOUT_NLC;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NHWC;
    } else if (5 == dim) {
        DIOPI_CHECK(interpMode != CNNL_INTERP_NEAREST, "nearest upsample for 5d tensor is not supported by camb kernel.");
        DIOPI_CHECK(inputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast3d");
        DIOPI_CHECK(outputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "outputTensor's memory format should be channelsLast3d");
        layout = CNNL_LAYOUT_NDHWC;
    } else {
        DIOPI_CHECK(false, "Dim of input tensor should be in [3,4,5].");
    }

    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTensor, layout);

    CnnlInterpDescriptor interpDesc;
    cnnlInterpCoordinateTransformationMode_t coordinateTransMode;
    if (!alignCorners && alignCenter) {
        coordinateTransMode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO0;
    } else if (alignCorners && !alignCenter) {
        coordinateTransMode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO2;
    } else if (!alignCorners && !alignCenter) {
        coordinateTransMode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO3;
    } else {
        DIOPI_CHECK(false, "unsupported combination of alignCorners and alignCenters");
    }

    DIOPI_CALL(interpDesc.set(inputDesc.get(), interpMode, coordinateTransMode, nullptr));

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALL_CNNL(cnnlInterp_v3(handle, interpDesc.get(), inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTensor.data()));

    return diopiSuccess;
}

diopiError_t upsampleBackwardInternal(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, bool alignCorners,
                                      bool alignCenter, cnnlInterpBackwardMode_t interpMode) {
    DiopiTensor inputTensor(gradOutput);
    DiopiTensor outputTensor(gradInput);

    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    auto dim = inputTensor.dim();
    cnnlTensorLayout_t layout;
    if (3 == dim) {
        DIOPI_CHECK(inputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NLC;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NHWC;
    } else if (5 == dim) {
        DIOPI_CHECK(inputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outputTensor.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NDHWC;
    } else {
        DIOPI_CHECK(false, "Dim of input tensor should be in [3,4,5].");
    }

    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTensor, layout);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALL_CNNL(cnnlInterpBackward_v2(
        handle, alignCorners, alignCenter, interpMode, nullptr, true, inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTensor.data()));

    return diopiSuccess;
}

diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    DIOPI_CALL(upsampleInternal(ctx, out, input, false, false, CNNL_INTERP_NEAREST))
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                          diopiSize_t inSize) {
    DIOPI_CALL(upsampleBackwardInternal(ctx, gradInput, gradOutput, false, false, CNNL_INTERP_BACKWARD_NEAREST))

    return diopiSuccess;
}

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool alignCorners,
                                 const char* mode) {
    DiopiTensor inputTensor(input);

    if (3 == inputTensor.dim() && 0 == strcmp(mode, "linear")) {
        DIOPI_CALL(upsampleInternal(ctx, out, input, alignCorners, !alignCorners, CNNL_INTERP_LINEAR));
    } else if (4 == inputTensor.dim()) {
        if (0 == strcmp(mode, "bilinear")) {
            DIOPI_CALL(upsampleInternal(ctx, out, input, alignCorners, !alignCorners, CNNL_INTERP_BILINEAR))
        } else if (0 == strcmp(mode, "bicubic")) {
            DIOPI_CALL(upsampleInternal(ctx, out, input, alignCorners, !alignCorners, CNNL_INTERP_BICUBIC))
        } else {
            DIOPI_CHECK(false, "interpolate mode type not supported");
            return diopiErrorOccurred;
        }
    } else if (5 == inputTensor.dim() && 0 == strcmp(mode, "trilinear")) {
        DIOPI_CALL(upsampleInternal(ctx, out, input, alignCorners, !alignCorners, CNNL_INTERP_TRILINEAR))
    } else {
        DIOPI_CHECK(false, "interpolate mode type not supported");
        return diopiErrorOccurred;
    }

    return diopiSuccess;
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                         diopiSize_t inSize, bool alignCorners, const char* mode) {
    DiopiTensor inputTensor(gradOutput);
    auto dim = inputTensor.dim();

    if (3 == dim && 0 == strcmp(mode, "linear")) {
        DIOPI_CALL(upsampleBackwardInternal(ctx, gradInput, gradOutput, alignCorners, !alignCorners, CNNL_INTERP_BACKWARD_LINEAR))
    } else if (4 == dim) {
        if (0 == strcmp(mode, "bilinear")) {
            DIOPI_CALL(upsampleBackwardInternal(ctx, gradInput, gradOutput, alignCorners, !alignCorners, CNNL_INTERP_BACKWARD_BILINEAR))
        } else if (0 == strcmp(mode, "bicubic")) {
            DIOPI_CALL(upsampleBackwardInternal(ctx, gradInput, gradOutput, alignCorners, !alignCorners, CNNL_INTERP_BACKWARD_BICUBIC))
        } else {
            DIOPI_CHECK(false, "interpolate mode type not supported.");
        }
    } else if (5 == dim && 0 == strcmp(mode, "trilinear")) {
        DIOPI_CALL(upsampleBackwardInternal(ctx, gradInput, gradOutput, alignCorners, !alignCorners, CNNL_INTERP_BACKWARD_TRILINEAR))
    } else {
        DIOPI_CHECK(false, "interpolate mode type not supported.");
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
