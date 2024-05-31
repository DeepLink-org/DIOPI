/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <algorithm>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    std::cout << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "calling diopiMaskedSelect" << std::endl;

    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);

    if (input == nullptr || inputAt.numel() == 0) {
        *out = nullptr;
        return diopiSuccess;
    }

    std::cout << "input dataFormat = " << inputAt.getAclDataFormat() << std::endl;
    std::cout << "mask dataFormat = " << maskAt.getAclDataFormat() << std::endl;

    // for aclnnMaskedSelect, the shape of out is one-dimensional,
    // with the number of elements equal to the broadcasted shape size of mask and self.
    // the shape of input and mask must be broadcastable.

    if (inputAt.dim() == 0) {
        inputAt.unsqueeze(0);
    }

    if (maskAt.dim() == 0) {
        maskAt.unsqueeze(0);
    }

    int64_t broadcastDim = std::max(inputAt.dim(), maskAt.dim());
    std::cout << "broadcastDim = " << broadcastDim << std::endl;

    while(inputAt.dim() < broadcastDim) {
        inputAt.unsqueeze(0);
    }

    while(maskAt.dim() < broadcastDim) {
        maskAt.unsqueeze(0);
    }

    int64_t broadcastShapeData[broadcastDim];
    int64_t outputNumel = 1;
    for (int64_t i = 0; i < broadcastDim; i++) {
        broadcastShapeData[i] = std::max(inputAt.shape(i), maskAt.shape(i));
        std::cout << "broadcastShapeData[" << i << "] = " << broadcastShapeData[i] << std::endl;
        outputNumel *= broadcastShapeData[i];
    }
    
    diopiSize_t outputShape{&outputNumel, 1};
    std::cout << "outputNumel = " << outputNumel << std::endl;

    diopiTensorHandle_t outTmp = nullptr;
    // makeTensorFromSize(ctx, &outputShape, &outTmp, inputAt.dtype());
    diopiRequireTensor(ctx, &outTmp, &outputShape, nullptr, inputAt.dtype(), diopi_device);

    AscendTensor outAt(outTmp);
    std::cout << "out dataFormat = " << outAt.getAclDataFormat() << std::endl;
    std::cout << "out.dim = " << outAt.dim() << std::endl;

    std::vector<int64_t> broadcastShape {broadcastShapeData, broadcastShapeData + broadcastDim};
    inputAt.view(broadcastShape);
    maskAt.view(broadcastShape);


    printVector(inputAt.shape());
    printVector(maskAt.shape());
    printVector(outAt.shape());

    std::cout << "FLAG 1" << std::endl;
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaskedSelect, ctx, inputAt, maskAt, outAt);
    std::cout << "FLAG 2" << std::endl;
    *out = outTmp;
    std::cout << "FLAG 3" << std::endl;
    
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);
    AscendTensor gradInputAt(gradInput);
    AscendTensor gradOutputAt(gradOutput);

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInput);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedScatter, ctx, gradInput, maskAt, gradOutputAt);


    
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
