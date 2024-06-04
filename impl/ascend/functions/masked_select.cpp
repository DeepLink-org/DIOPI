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
diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);

    if (input == nullptr || inputAt.numel() == 0 || mask == nullptr || maskAt.numel() == 0) {
        int64_t zero = 0;
        diopiSize_t emptyShape{&zero, 0};
        diopiRequireTensor(ctx, out, &emptyShape, nullptr, inputAt.dtype(), diopi_device);
        return diopiSuccess;
    }

    // calculate the broadcastShape of inputAt and maskAt and the number of elements
    if (inputAt.dim() == 0) {
        inputAt.unsqueeze(0);
    }

    if (maskAt.dim() == 0) {
        maskAt.unsqueeze(0);
    }

    int64_t broadcastDim = std::max(inputAt.dim(), maskAt.dim());

    while (inputAt.dim() < broadcastDim) {
        inputAt.unsqueeze(0);
    }

    while (maskAt.dim() < broadcastDim) {
        maskAt.unsqueeze(0);
    }

    int64_t broadcastShapeData[broadcastDim];
    int64_t broadcastNumel = 1;
    for (int64_t i = 0; i < broadcastDim; i++) {
        broadcastShapeData[i] = std::max(inputAt.shape(i), maskAt.shape(i));
        broadcastNumel *= broadcastShapeData[i];
    }

    diopiTensorHandle_t outTmp = nullptr;
    diopiTensorHandle_t nonZero = nullptr;
    diopiTensorHandle_t maskBroadcast = nullptr;
    diopiSize_t broadcastShape{broadcastShapeData, broadcastDim};

    // make maskBroadCast Tensor and calculate the number of non-zero elements
    diopiRequireTensor(ctx, &maskBroadcast, &broadcastShape, nullptr, diopi_dtype_bool, diopi_device);
    diopiExpand(ctx, maskBroadcast, mask);
    diopiNonzero(ctx, &nonZero, maskBroadcast);

    // The actual number of output elements is the number of non-zero elements in the mask.
    AscendTensor nonZeroAt(nonZero);
    int64_t nonZeroNumel = nonZeroAt.shape(0);
    diopiSize_t outputShape{&nonZeroNumel, 1};
    diopiRequireTensor(ctx, &outTmp, &outputShape, nullptr, inputAt.dtype(), diopi_device);

    // reshape the AscendTensor outAt because aclnnMaskedSelect limitation
    // for aclnnMaskedSelect, the shape of out is one-dimensional,
    // with the number of elements equal to the broadcasted shape size of mask and self.
    // the shape of input and mask must be broadcastable.
    AscendTensor outAt(outTmp);
    outAt.view({broadcastNumel});
    DIOPI_ASCEND_CALL_ACLNN(aclnnMaskedSelect, ctx, inputAt, maskAt, outAt);
    *out = outTmp;

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);
    AscendTensor gradInputAt(gradInput);
    AscendTensor gradOutputAt(gradOutput);

    if (input == nullptr || inputAt.numel() == 0 || mask == nullptr || maskAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInput);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedScatter, ctx, gradInput, maskAt, gradOutputAt);

    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
