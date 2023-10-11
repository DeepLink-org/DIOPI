/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <cmath>
#include <limits>
#include <string>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool alignCorners,
                                 const char* mode) {
    std::string modeStr(mode);
    if ("bilinear" == modeStr) {
        AclOpRunner<2, 1>("ResizeBilinearV2", ctx)
            .addInput(input)
            .addConstInput(size)
            .setAttr("align_corners", alignCorners)
            .setAttr("half_pixel_centers", !alignCorners)
            .addOutput(out)
            .run();
    } else {
        ASCEND_CHECK_ABORT(false, "unsupport mode %s", modeStr.c_str());
    }
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                         diopiSize_t inSize, bool alignCorners, const char* mode) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(gradOutput, &dtype);
    diopiSize_t stride;
    diopiGetTensorStride(gradOutput, &stride);
    diopiTensorHandle_t originalImage;
    diopiRequireTensor(ctx, &originalImage, &inSize, nullptr, dtype, diopi_device);
    std::string modeStr(mode);
    if ("bilinear" == modeStr) {
        AclOpRunner<2, 1>("ResizeBilinearV2Grad", ctx)
            .addInput(gradOutput)
            .addInput(originalImage)
            .setAttr("align_corners", alignCorners)
            .setAttr("half_pixel_centers", !alignCorners)
            .addOutput(gradInput)
            .run();
    } else {
        ASCEND_CHECK_ABORT(false, "unsupport mode %s", modeStr.c_str());
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
