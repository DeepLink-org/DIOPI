/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnRelu, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceRelu, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    AscendTensor inputTensor(input);
    if (inputTensor.dim() == 0) {
        diopiScalar_t value = constructDiopiScalarT(inputTensor.dtype(), 1.0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, out, &value);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnSoftmax, ctx, input, dim, out);
    }
    return diopiSuccess;
}

diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t output,
                                  int64_t dim) {
    AscendTensor gradInputTensor(gradInput);
    if (gradInputTensor.dim() == 0) {
        diopiScalar_t value = constructDiopiScalarT(gradInputTensor.dtype(), 0.0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, gradInput, &value);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnSoftmaxBackward, ctx, gradOutput, output, dim, gradInput);
    }
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    AscendTensor inputTensor(input);
    if (inputTensor.dim() == 0) {
        diopiScalar_t value = constructDiopiScalarT(inputTensor.dtype(), 0.0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, out, &value);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnLogSoftmax, ctx, input, dim, out);
    }
    return diopiSuccess;
}

diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                     diopiConstTensorHandle_t output, int64_t dim) {
    AscendTensor gradInputTensor(gradInput);
    if (gradInputTensor.dim() == 0) {
        diopiScalar_t value = constructDiopiScalarT(gradInputTensor.dtype(), 0.0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, gradInput, &value);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnLogSoftmaxBackward, ctx, gradOutput, output, dim, gradInput);
    }
    return diopiSuccess;
}

diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSilu, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSilu, ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSiluBackward, ctx, gradOutput, input, gradInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSigmoid, ctx, input, out);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceSigmoid, ctx, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t output) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSigmoidBackward, ctx, gradOutput, output, gradInput);
    return diopiSuccess;
}

diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnGelu, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                               const char* approximate) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnGeluBackward, ctx, gradOutput, input, gradInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLeakyRelu, ctx, input, negativeSlope, out);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLeakyRelu, ctx, input, negativeSlope);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope, bool inputIsResult) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLeakyReluBackward, ctx, gradOutput, input, negativeSlope, inputIsResult, gradInput);
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnTanh, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceTanh, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t output) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnTanhBackward, ctx, gradOutput, output, gradInput);
    return diopiSuccess;
}

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minVal,
                           const diopiScalar_t* maxVal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnHardtanh, ctx, input, minVal, maxVal, out);
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceHardtanh, ctx, input, minVal, maxVal);
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                   const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnHardtanhBackward, ctx, gradOutput, input, minVal, maxVal, gradInput);
    return diopiSuccess;
}

diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnHardswish, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceHardswish, ctx, input);
    return diopiSuccess;
}

diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnHardswishBackward, ctx, gradOutput, input, gradInput);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
