/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    // outputPadding will be used when transposed is True
    bool transposed = false;
    // TODO(zhangqiu) impl int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
    int8_t cubeMathType = 0;

    ASCEND_CHECK_ABORT(stride.len == 1 || stride.len == 2, "the dim of stride must be 1 or 2!");
    ASCEND_CHECK_ABORT(padding.len == 1 || padding.len == 2, "the dim of padding must be 1 or 2!");
    ASCEND_CHECK_ABORT(dilation.len == 1 || dilation.len == 2, "the dim of dilation must be 1 or 2!");

    int64_t outputPaddingData[2];
    for (int i = 0; i < padding.len; i++) {
        outputPaddingData[i] = 0;
    }
    diopiSize_t outputPadding{outputPaddingData, 2};

    int64_t strideExpandData[2];
    int64_t paddingExpandData[2];
    int64_t dilationExpandData[2];

    strideExpandData[0] = stride.data[0];
    strideExpandData[1] = (stride.len == 1) ? stride.data[0] : stride.data[1];

    paddingExpandData[0] = padding.data[0];
    paddingExpandData[1] = (padding.len == 1) ? padding.data[0] : padding.data[1];

    dilationExpandData[0] = dilation.data[0];
    dilationExpandData[1] = (dilation.len == 1) ? dilation.data[0] : dilation.data[1];

    DIOPI_ASCEND_CALL_ACLNN(aclnnConvolution,
                            ctx,
                            input,
                            weight,
                            bias,
                            diopiSize_t{strideExpandData, 2},
                            diopiSize_t{paddingExpandData, 2},
                            diopiSize_t{dilationExpandData, 2},
                            transposed,
                            outputPadding,
                            groups,
                            out,
                            cubeMathType);
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

    int64_t outputPaddingData[2];
    for (int i = 0; i < padding.len; i++) {
        outputPaddingData[i] = 0;
    }
    diopiSize_t outputPadding{outputPaddingData, 2};

    int64_t strideExpandData[2];
    int64_t paddingExpandData[2];
    int64_t dilationExpandData[2];

    strideExpandData[0] = stride.data[0];
    strideExpandData[1] = (stride.len == 1) ? stride.data[0] : stride.data[1];

    paddingExpandData[0] = padding.data[0];
    paddingExpandData[1] = (padding.len == 1) ? padding.data[0] : padding.data[1];

    dilationExpandData[0] = dilation.data[0];
    dilationExpandData[1] = (dilation.len == 1) ? dilation.data[0] : dilation.data[1];

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
                            diopiSize_t{strideExpandData, 2},
                            diopiSize_t{paddingExpandData, 2},
                            diopiSize_t{dilationExpandData, 2},
                            transposed,
                            outputPadding,
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
