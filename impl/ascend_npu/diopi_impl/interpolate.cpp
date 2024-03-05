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
    std::string modeStr(mode);
    TORCH_CHECK(modeStr == "bilinear", "diopiUpsampleLinearBackward unsupport mode %s", modeStr.c_str());
    BEGIN_CALL_ACL_OP(input, out);
    std::vector<int64_t> sizeVec(size.data, size.data + size.len);
    double scalesH = 1.0;
    double scalesW = 1.0;
    op_api::upsample_bilinear2d_out(inputAt, sizeVec, alignCorners, scalesH, scalesW, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t outSize,
                                         diopiSize_t inSize, bool alignCorners, const char* mode) {
    std::string modeStr(mode);
    TORCH_CHECK(modeStr == "bilinear", "diopiUpsampleLinearBackward unsupport mode %s", modeStr.c_str());
    BEGIN_CALL_ACL_OP(gradInput, gradOutput);
    std::vector<int64_t> outSizeVec(outSize.data, outSize.data + outSize.len);
    std::vector<int64_t> inSizeVec(inSize.data, inSize.data + inSize.len);
    double scalesH = 1.0;
    double scalesW = 1.0;
    op_api::upsample_bilinear2d_backward_out(gradOutputAt, outSizeVec, inSizeVec, alignCorners, scalesH, scalesW, gradInputAt);
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
