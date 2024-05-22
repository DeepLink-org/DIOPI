/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include<algorithm>

namespace impl {
namespace ascend {
    diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    
    std::cout << std::endl;
    std::cout << "calling diopiMaskedSelect" << std::endl;
    
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);

    // for aclnnMaskedSelect, the shape of out is one-dimensional, 
    // with the number of elements equal to the broadcasted shape size of mask and self.
    // the shape of input and mask must be broadcastable.
    int64_t broadcastDim = std::max(inputAt.dim(), maskAt.dim());
    int64_t broadcastShapeData[broadcastDim];
    int64_t outputNumel = 1;
    for (int64_t i = 0; i < broadcastDim; i++) {
        broadcastShapeData[i] = std::max(inputAt.shape(i), maskAt.shape(i));
        std::cout << "broadcastShapeData[i] = " << broadcastShapeData[i] << std::endl;
        outputNumel *= broadcastShapeData[i];
    }


    diopiSize_t broadcastShape{broadcastShapeData, broadcastDim};
    diopiSize_t outputShape{&outputNumel, 1};
    diopiTensorHandle_t inputBroadcast;
    diopiTensorHandle_t maskBroadcast;
    diopiTensorHandle_t outTmp;
    makeTensorFromSize(ctx, &broadcastShape, &inputBroadcast, inputAt.dtype());
    makeTensorFromSize(ctx, &broadcastShape, &maskBroadcast, diopi_dtype_bool);
    std::cout << "flag 1" << std::endl;
    DIOPI_ASCEND_CALL_ACLNN(aclnnExpand, ctx, input, inputBroadcast, inputBroadcast);
    std::cout << "flag 2" << std::endl;
    DIOPI_ASCEND_CALL_ACLNN(aclnnExpand, ctx, mask, broadcastShape, maskBroadcast);
    makeTensorFromSize(ctx, &outputShape, &outTmp, inputAt.dtype());

    DIOPI_ASCEND_CALL_ACLNN(aclnnMaskedSelect, ctx, inputBroadcast, maskBroadcast, outTmp);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, *out, outTmp);
    return diopiSuccess;
}

    DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
                                                    return diopiSuccess;
                                                 }
}  // namespace camb
}  // namespace impl