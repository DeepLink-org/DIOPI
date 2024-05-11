/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    bool transposed = false;
    // TODO(zhangqiu) impl int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
    int8_t cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnConvolution, ctx, input, weight, bias, stride, padding, dilation, transposed, nullptr, groups, cubeMathType, out);
    return diopiSuccess;
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                        diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                        diopiSize_t* biasSizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    bool transposed = false;
    // TODO(zhangqiu) impl int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
    int8_t cubeMathType = 0;
    int64_t gradMask[3] = {true, true, true};
    diopiSize_t outputMask{gradMask, 3};
    DIOPI_ASCEND_CALL_ACLNN(aclnnConvolutionBackward,
                            ctx,
                            gradOutput,
                            input,
                            weight,
                            *biasSizes,
                            stride,
                            padding,
                            dilation,
                            transposed,
                            nullptr,
                            groups,
                            outputMask,
                            cubeMathType,
                            gradInput,
                            gradWeight,
                            gradBias);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
