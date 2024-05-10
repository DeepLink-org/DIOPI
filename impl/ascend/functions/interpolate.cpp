/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool alignCorners,
                                 const char* mode) {
    ASCEND_CHECK(strcmp(mode, "bilinear") == 0, "diopiUpsampleLinearBackward unsupport mode %s", mode);
    DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleBilinear2d, ctx, input, size, alignCorners, 1.0, 1.0, out);
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                         diopiSize_t inSize, bool alignCorners, const char* mode) {
    ASCEND_CHECK(strcmp(mode, "bilinear") == 0, "diopiUpsampleLinearBackward unsupport mode %s", mode);
    DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleBilinear2dBackward, ctx, gradOutput, outSize, inSize, alignCorners, 1.0, 1.0, gradInput);
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    if (1 == size.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest1d, ctx, input, size, out);
    } else if (2 == size.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest2d, ctx, input, size, out);
    } else if (3 == size.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest3d, ctx, input, size, 0, 0, 0, out);
    }
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                          diopiSize_t inSize) {
    if (1 == outSize.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest1dBackward, ctx, gradOutput, outSize, inSize, -1.0, gradInput);
    } else if (2 == outSize.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest2dBackward, ctx, gradOutput, outSize, inSize, -1.0, -1.0, gradInput);
    } else if (3 == outSize.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnUpsampleNearest3dBackward, ctx, gradOutput, outSize, inSize, 0, 0, 0, gradInput);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
