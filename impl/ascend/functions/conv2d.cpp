/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

static diopiSize_t expandDim(diopiSize_t inputShape, int64_t expectedDim) {
    if (inputShape.len == 1) {
        int64_t expandShapeData[expectedDim];
        for (int64_t i = 0; i < expectedDim; i++) {
            expandShapeData[i] = inputShape.data[0];
        }
        diopiSize_t expandShape{expandShapeData, expectedDim};
        return expandShape;
    } else {
        return inputShape;
    }
}
namespace impl {
namespace ascend {

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    // outputPadding will be used when transposed is True
    bool transposed = false;
    // TODO(zhangqiu) impl int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
    int8_t cubeMathType = 0;

    int64_t outputPaddingData[padding.len];
    for (int i = 0; i < padding.len; i++) {
        outputPaddingData[i] = 0;
    }
    diopiSize_t outputPadding{outputPaddingData, padding.len};

    auto strideExpand = expandDim(stride, 2);
    auto paddingExpand = expandDim(padding, 2);
    auto dilationExpand = expandDim(dilation, 2);
    auto outputPaddingExpand = expandDim(outputPadding, 2);

    DIOPI_ASCEND_CALL_ACLNN(
        aclnnConvolution, ctx, input, weight, bias, strideExpand, paddingExpand, dilationExpand, transposed, outputPaddingExpand, groups, out, cubeMathType);
    return diopiSuccess;
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                        diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                        diopiSize_t* biasSizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    bool transposed = false;
    // TODO(zhangqiu) impl int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
    int8_t cubeMathType = 0;

    std::array<bool, 3> gradMask = {true, true, true};
    if (nullptr == gradInput) {
        gradMask[0] = false;
    }
    if (nullptr == gradWeight) {
        gradMask[1] = false;
    }
    if (nullptr == gradBias) {
        gradMask[2] = false;
    }

    int64_t outputPaddingData[padding.len];
    for (int i = 0; i < padding.len; i++) {
        outputPaddingData[i] = 0;
    }
    diopiSize_t outputPadding{outputPaddingData, padding.len};

    auto strideExpand = expandDim(stride, 2);
    auto paddingExpand = expandDim(padding, 2);
    auto dilationExpand = expandDim(dilation, 2);
    auto outputPaddingExpand = expandDim(outputPadding, 2);

    AscendTensor gradBiasAt(gradBias);
    std::vector<int64_t> biasShape;
    if (gradBias != nullptr) {
        biasShape = gradBiasAt.shape();
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnConvolutionBackward,
                            ctx,
                            gradOutput,
                            input,
                            weight,
                            biasShape,
                            strideExpand,
                            paddingExpand,
                            dilationExpand,
                            transposed,
                            outputPaddingExpand,
                            groups,
                            gradMask,
                            cubeMathType,
                            gradInput,
                            gradWeight,
                            gradBias);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
