/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <c10/core/ScalarType.h>

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool alignCorners,
                                 const char* mode) {
    TORCH_CHECK(strcmp(mode, "bilinear") == 0, "diopiUpsampleLinearBackward unsupport mode %s", mode);
    BEGIN_CALL_ACL_OP(input, out);
    std::vector<int64_t> sizeVec(size.data, size.data + size.len);
    double scalesH = 1.0;
    double scalesW = 1.0;
    op_api::upsample_bilinear2d_out(inputAt, sizeVec, alignCorners, scalesH, scalesW, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                         diopiSize_t inSize, bool alignCorners, const char* mode) {
    TORCH_CHECK(strcmp(mode, "bilinear") == 0, "diopiUpsampleLinearBackward unsupport mode %s", mode);
    BEGIN_CALL_ACL_OP(gradInput, gradOutput);
    std::vector<int64_t> outSizeVec(outSize.data, outSize.data + outSize.len);
    std::vector<int64_t> inSizeVec(inSize.data, inSize.data + inSize.len);
    double scalesH = 1.0;
    double scalesW = 1.0;
    op_api::upsample_bilinear2d_backward_out(gradOutputAt, outSizeVec, inSizeVec, alignCorners, scalesH, scalesW, gradInputAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    BEGIN_CALL_ACL_OP(input, out);
    std::vector<int64_t> sizeVec(size.data, size.data + size.len);
    double scalesH = 1.0;
    double scalesW = 1.0;
    double scalesD = 1.0;
    if (sizeVec.size() == 1) {
        op_api::upsample_nearest1d_out(inputAt, sizeVec, scalesD, outAt);
    } else if (sizeVec.size() == 2) {
        op_api::upsample_nearest2d_out(inputAt, sizeVec, scalesH, scalesW, outAt);
    } else if (sizeVec.size() == 3) {
        op_api::upsample_nearest3d_out(inputAt, sizeVec, scalesD, scalesW, scalesH, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                          diopiSize_t inSize) {
    BEGIN_CALL_ACL_OP(gradInput, gradOutput);
    std::vector<int64_t> outSizeVec(outSize.data, outSize.data + outSize.len);
    std::vector<int64_t> inSizeVec(inSize.data, inSize.data + inSize.len);
    double scalesH = 1.0;
    double scalesW = 1.0;
    double scalesD = 1.0;
    if (outSizeVec.size() == 1) {
        scalesD = gradOutputAt.size(2) * 1.0 / gradInputAt.size(2);
        op_api::upsample_nearest1d_backward_out(gradOutputAt, outSizeVec, inSizeVec, scalesD, gradInputAt);
    } else if (outSizeVec.size() == 2) {
        op_api::upsample_nearest2d_backward_out(gradOutputAt, outSizeVec, inSizeVec, scalesH, scalesW, gradInputAt);
    } else if (outSizeVec.size() == 3) {
        op_api::upsample_nearest3d_backward_out(gradOutputAt, outSizeVec, inSizeVec, scalesD, scalesH, scalesW, gradInputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
