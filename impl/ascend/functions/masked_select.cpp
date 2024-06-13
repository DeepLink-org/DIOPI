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

    // broadcast input and mask
    maskAt.view(std::vector<int64_t>(broadcastShape.data, broadcastShape.data + broadcastDim));
    inputAt.view(std::vector<int64_t>(broadcastShape.data, broadcastShape.data + broadcastDim));

    diopiRequireTensor(ctx, &outTmp, &broadcastShape, nullptr, inputAt.dtype(), diopi_device);
    AscendTensor outTmpAt(outTmp);

    auto params = DIOPI_ASECND_CALL_ACLNN_SYNC(aclnnMaskedSelect, ctx, inputAt, maskAt, outTmp);
    int64_t *viewDims = nullptr;
    uint64_t viewDimNum = 0;
    int ret = aclGetViewShape(std::get<2>(params.params()), &viewDims, &viewDimNum);
    ASCEND_CHECK_ABORT(ret == 0, "aclGetViewShape failed");
    std::vector<int64_t> outputShapeSize;
    for (uint64_t i = 0; i < viewDimNum; i++) {
        outputShapeSize.push_back(viewDims[i]);
    }

    outTmpAt.view(outputShapeSize);
    *out = outTmp;
    delete viewDims;

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
