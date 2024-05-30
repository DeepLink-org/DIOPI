/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

void maxPool2dCheck(const AscendTensor& input, diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation) {
    ASCEND_CHECK_ABORT(kernelSize.len == 1 || kernelSize.len == 2, "max_pool2d: kernel_size must either be a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT(stride.len == 0 || stride.len == 1 || stride.len == 2,
                       "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT(padding.len == 1 || padding.len == 2, "max_pool2d: padding must be either be a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT(dilation.len == 1 || dilation.len == 2, "max_pool2d: dilation must be either a single int, or a tuple of two ints");
    ASCEND_CHECK_ABORT(input.dim() == 3 || input.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");
}

diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    int64_t kH = kernelSize.data[0];
    int64_t kW = kernelSize.len == 1 ? kH : kernelSize.data[1];
    int64_t kernelSizeData[2]{kH, kW};

    int64_t sH = stride.len == 0 ? kH : stride.data[0];
    int64_t sW = stride.len == 0 ? kW : (stride.len == 1 ? sH : stride.data[1]);
    int64_t strideData[2]{sH, sW};

    int64_t padH = padding.data[0];
    int64_t padW = padding.len == 1 ? padH : padding.data[1];
    int64_t paddingData[2]{padH, padW};

    int64_t dH = dilation.data[0];
    int64_t dW = dilation.len == 1 ? dH : dilation.data[1];
    int64_t dilationData[2]{dH, dW};

    int64_t nBatch = inputAt.dim() == 4 ? inputAt.shape(-4) : 1;
    int64_t nInputPlane = inputAt.shape(-3);
    int64_t inputHeight = inputAt.shape(-2);
    int64_t inputWidth = inputAt.shape(-1);

    const int64_t blockSize = 16;
    int64_t outHeight = outAt.shape(-2);
    int64_t outWidth = outAt.shape(-1);

    int64_t maskH = kernelSizeData[0] * kernelSizeData[1];
    int64_t maskW = (outHeight * outWidth) / blockSize + 1;

    std::vector<int64_t> indicesShape;
    if (inputAt.dim() == 4) {
        indicesShape = std::vector<int64_t>{nBatch, nInputPlane, maskH, maskW * 32};
    } else {
        indicesShape = std::vector<int64_t>{nInputPlane, maskH, maskW * 32};
    }

    maxPool2dCheck(inputAt, diopiSize_t{kernelSizeData, 2}, diopiSize_t{strideData, 2}, diopiSize_t{paddingData, 2}, diopiSize_t{dilationData, 2});

    AscendTensor indicesAt;
    makeTensor(ctx, indicesAt, indicesShape, diopi_dtype_int8);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxPool2dWithMask,
                            ctx,
                            inputAt,
                            diopiSize_t{kernelSizeData, 2},
                            diopiSize_t{strideData, 2},
                            diopiSize_t{paddingData, 2},
                            diopiSize_t{dilationData, 2},
                            ceilMode,
                            outAt,
                            indicesAt);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    AscendTensor inputAt(input);
    maxPool2dCheck(inputAt, kernelSize, stride, padding, dilation);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxPool2dWithMask, ctx, input, kernelSize, stride, padding, dilation, ceilMode, out, indices);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceilMode, diopiConstTensorHandle_t indices) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxPool2dWithMaskBackward, ctx, gradOutput, input, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
