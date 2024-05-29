/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    std::cout << std::endl;
    std::cout << "calling diopiMaxPool2d" << std::endl;
    AscendTensor inputAt(input);
    AscendTensor outAt(out);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    if (inputAt.dim() == 3) {
        inputAt.unsqueeze(0);
        outAt.unsqueeze(0);
    }

    std::cout << "inputAt.dtype = " << inputAt.dtype() << std::endl;
    std::cout << "inputAt.dim = " << inputAt.dim() << std::endl;
    std::cout << "the dataFormat of input is " << inputAt.getAclDataFormat() << std::endl;

    int64_t kernelSizeExpandData[2];
    int64_t strideExpandData[2];
    int64_t paddingExpandData[2];
    int64_t dilationExpandData[2];

    kernelSizeExpandData[0] = kernelSize.data[0];
    kernelSizeExpandData[1] = (kernelSize.len == 1) ? kernelSize.data[0] : kernelSize.data[1];

    std::cout << "stride.len = " << stride.len << std::endl;
    std::cout << "stride.data = ";
    for (int i = 0; i < stride.len; i++) {
        std::cout << stride.data[i] << " ";
    }
    std::cout << std::endl;
    if (stride.len == 0) {
        strideExpandData[0] = kernelSizeExpandData[0];
        kernelSizeExpandData[1] = strideExpandData[1];
    } else {
        strideExpandData[0] = stride.data[0];
        strideExpandData[1] = (stride.len == 1) ? stride.data[0] : stride.data[1];
    }

    paddingExpandData[0] = padding.data[0];
    paddingExpandData[1] = (padding.len == 1) ? padding.data[0] : padding.data[1];

    dilationExpandData[0] = dilation.data[0];
    dilationExpandData[1] = (dilation.len == 1) ? dilation.data[0] : dilation.data[1];
    ASCEND_CHECK_ABORT(dilationExpandData[0] == 1 and dilationExpandData[1] == 1, "aclnnMaxPool kenel only support dilaton_value = 1 for now.");

    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxPool,
                            ctx,
                            inputAt,
                            diopiSize_t{kernelSizeExpandData, 2},
                            diopiSize_t{strideExpandData, 2},
                            0,
                            diopiSize_t{paddingExpandData, 2},
                            diopiSize_t{dilationExpandData, 2},
                            ceilMode,
                            outAt);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceilMode) {
    std::cout << std::endl;
    std::cout << "calling diopiMaxPool2dWithIndices" << std::endl;
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    int64_t kernelSizeExpandData[2];
    int64_t strideExpandData[2];
    int64_t paddingExpandData[2];
    int64_t dilationExpandData[2];

    kernelSizeExpandData[0] = kernelSize.data[0];
    kernelSizeExpandData[1] = (kernelSize.len == 1) ? kernelSize.data[0] : kernelSize.data[1];

    if (stride.len == 0) {
        strideExpandData[0] = kernelSizeExpandData[0];
        kernelSizeExpandData[1] = strideExpandData[1];
    } else {
        strideExpandData[0] = stride.data[0];
        strideExpandData[1] = (stride.len == 1) ? stride.data[0] : stride.data[1];
    }

    paddingExpandData[0] = padding.data[0];
    paddingExpandData[1] = (padding.len == 1) ? padding.data[0] : padding.data[1];

    dilationExpandData[0] = dilation.data[0];
    dilationExpandData[1] = (dilation.len == 1) ? dilation.data[0] : dilation.data[1];
    ASCEND_CHECK_ABORT(dilationExpandData[0] == 1 and dilationExpandData[1] == 1, "aclnnMaxPool kenel only support dilaton_value = 1 for now.");

    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxPool2dWithIndices,
                            ctx,
                            input,
                            diopiSize_t{kernelSizeExpandData, 2},
                            diopiSize_t{strideExpandData, 2},
                            diopiSize_t{paddingExpandData, 2},
                            diopiSize_t{dilationExpandData, 2},
                            ceilMode,
                            out,
                            indices);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                    diopiConstTensorHandle_t input, diopiSize_t kernelSize, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceilMode, diopiConstTensorHandle_t indices) {
    std::cout << std::endl;
    std::cout << "callong diopiMaxPool2dBackward" << std::endl;
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    int64_t kernelSizeExpandData[2];
    int64_t strideExpandData[2];
    int64_t paddingExpandData[2];
    int64_t dilationExpandData[2];

    kernelSizeExpandData[0] = kernelSize.data[0];
    kernelSizeExpandData[1] = (kernelSize.len == 1) ? kernelSize.data[0] : kernelSize.data[1];

    if (stride.len == 0) {
        strideExpandData[0] = kernelSizeExpandData[0];
        kernelSizeExpandData[1] = strideExpandData[1];
    } else {
        strideExpandData[0] = stride.data[0];
        strideExpandData[1] = (stride.len == 1) ? stride.data[0] : stride.data[1];
    }

    paddingExpandData[0] = padding.data[0];
    paddingExpandData[1] = (padding.len == 1) ? padding.data[0] : padding.data[1];

    dilationExpandData[0] = dilation.data[0];
    dilationExpandData[1] = (dilation.len == 1) ? dilation.data[0] : dilation.data[1];
    ASCEND_CHECK_ABORT(dilationExpandData[0] == 1 and dilationExpandData[1] == 1, "aclnnMaxPool kenel only support dilaton_value = 1 for now.");

    DIOPI_ASCEND_CALL_ACLNN(aclnnMaxPool2dWithIndices,
                            ctx,
                            gradOutput,
                            input,
                            indices,
                            diopiSize_t{kernelSizeExpandData, 2},
                            diopiSize_t{strideExpandData, 2},
                            diopiSize_t{paddingExpandData, 2},
                            diopiSize_t{dilationExpandData, 2},
                            ceilMode,
                            gradInput,
                            indices);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
