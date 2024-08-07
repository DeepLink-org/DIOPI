/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

// NOLINTNEXTLINE
constexpr double DEFAULT_SCALES = -1.0;

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool alignCorners,
                                 const char* mode) {
    if (0 == strcmp(mode, "bilinear")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleBilinear2d, ctx, input, size, alignCorners, DEFAULT_SCALES, DEFAULT_SCALES, out);
    } else if (0 == strcmp(mode, "bicubic")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleBicubic2d, ctx, input, size, alignCorners, DEFAULT_SCALES, DEFAULT_SCALES, out);
    } else if (0 == strcmp(mode, "trilinear")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleTrilinear3d, ctx, input, size, alignCorners, DEFAULT_SCALES, DEFAULT_SCALES, DEFAULT_SCALES, out);
    } else if (0 == strcmp(mode, "linear")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleLinear1d, ctx, input, size, alignCorners, DEFAULT_SCALES, out);
    } else {
        ASCEND_CHECK_ABORT(false, "unsupport mode %s", mode);
    }
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                         diopiSize_t inSize, bool alignCorners, const char* mode) {
    if (0 == strcmp(mode, "bilinear")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleBilinear2dBackward, ctx, gradOutput, outSize, inSize, alignCorners, DEFAULT_SCALES, DEFAULT_SCALES, gradInput);
    } else if (0 == strcmp(mode, "bicubic")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleBicubic2dBackward, ctx, gradOutput, outSize, inSize, alignCorners, DEFAULT_SCALES, DEFAULT_SCALES, gradInput);
    } else if (0 == strcmp(mode, "trilinear")) {
        DIOPI_ASCEND_CALL_ACLNN(
            aclnnUpsampleTrilinear3dBackward, ctx, gradOutput, outSize, inSize, alignCorners, DEFAULT_SCALES, DEFAULT_SCALES, DEFAULT_SCALES, gradInput);
    } else if (0 == strcmp(mode, "linear")) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleLinear1dBackward, ctx, gradOutput, outSize, inSize, alignCorners, DEFAULT_SCALES, gradInput);
    } else {
        ASCEND_CHECK_ABORT(false, "unsupport mode %s", mode);
    }
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    if (1 == size.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest1d, ctx, input, size, out);
    } else if (2 == size.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest2d, ctx, input, size, out);
    } else if (3 == size.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest3d, ctx, input, size, out);
    }
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                          diopiSize_t inSize) {
    if (1 == outSize.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest1dBackward, ctx, gradOutput, outSize, inSize, DEFAULT_SCALES, gradInput);
    } else if (2 == outSize.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest2dBackward, ctx, gradOutput, outSize, inSize, DEFAULT_SCALES, DEFAULT_SCALES, gradInput);
    } else if (3 == outSize.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest3dBackward, ctx, gradOutput, outSize, inSize, DEFAULT_SCALES, DEFAULT_SCALES, DEFAULT_SCALES, gradInput);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
