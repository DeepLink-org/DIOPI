/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <algorithm>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {
diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);

    // handle the corner cases
    if (input == nullptr || inputAt.numel() == 0 || mask == nullptr || maskAt.numel() == 0) {
        int64_t zero = 0;
        diopiSize_t emptyShape{&zero, 0};
        diopiRequireTensor(ctx, out, &emptyShape, nullptr, inputAt.dtype(), diopi_device);
        return diopiSuccess;
    }

    // calculate the broadcastShape of inputAt and maskAt, and the number of elements
    if (inputAt.dim() == 0) {
        inputAt = inputAt.unsqueeze(0);
    }

    if (maskAt.dim() == 0) {
        maskAt = maskAt.unsqueeze(0);
    }

    int64_t broadcastDim = std::max(inputAt.dim(), maskAt.dim());

    while (inputAt.dim() < broadcastDim) {
        inputAt = inputAt.unsqueeze(0);
    }

    while (maskAt.dim() < broadcastDim) {
        maskAt = maskAt.unsqueeze(0);
    }

    int64_t broadcastShapeData[broadcastDim];
    int64_t broadcastNumel = 1;
    for (int64_t i = 0; i < broadcastDim; i++) {
        broadcastShapeData[i] = std::max(inputAt.shape(i), maskAt.shape(i));
        broadcastNumel *= broadcastShapeData[i];
    }

    // broadcast input and mask
    diopiSize_t broadcastShape{broadcastShapeData, broadcastDim};
    diopiSize_t outShapeTmp{&broadcastNumel, 1};

    std::vector<int64_t> inputStride{inputAt.stride()};
    std::vector<int64_t> maskStride{maskAt.stride()};

    for (int64_t i = 0; i < broadcastDim; i++) {
        if (broadcastShape.data[i] != inputAt.shape(i)) {
            inputStride[i] = 0;
        }
        if (broadcastShape.data[i] != maskAt.shape(i)) {
            maskStride[i] = 0;
        }
    }

    AscendTensor expandInputAt = inputAt.asStrided({broadcastShapeData, broadcastShapeData + broadcastDim}, inputStride);
    AscendTensor expandMaskAt = maskAt.asStrided({broadcastShapeData, broadcastShapeData + broadcastDim}, maskStride);

    // call aclnnMaskedSelect to do the calculation
    diopiTensorHandle_t outTmp = nullptr;
    diopiRequireTensor(ctx, &outTmp, &outShapeTmp, nullptr, inputAt.dtype(), diopi_device);
    auto params = DIOPI_ASECND_CALL_ACLNN_SYNC(aclnnMaskedSelect, ctx, expandInputAt, expandMaskAt, outTmp);

    // get true outShape by aclGetViewShape
    int64_t* viewDims = nullptr;
    uint64_t viewDimNum = 0;
    using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
    aclGetViewShapeFunc aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(impl::ascend::aclnn_adaptor::getOpApiFuncAddr("aclGetViewShape"));
    int ret = aclGetViewShape(std::get<2>(params.params()), &viewDims, &viewDimNum);
    ASCEND_CHECK_ABORT(ret == 0, "aclGetViewShape failed");
    diopiSize_t outShape{viewDims, static_cast<int64_t>(viewDimNum)};

    // require out tensor from true outShape
    diopiRequireTensor(ctx, out, &outShape, nullptr, inputAt.dtype(), diopi_device);

    // copy outTmp to out
    AscendTensor outAt(*out);
    AscendTensor outTmpAt(outTmp);
    outTmpAt.view({outShape.data, outShape.data + outShape.len});
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, outTmpAt);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradout,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);
    AscendTensor gradInputAt(gradInput);
    AscendTensor gradoutAt(gradout);

    if (input == nullptr || inputAt.numel() == 0 || mask == nullptr || maskAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInput);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedScatter, ctx, gradInput, maskAt, gradoutAt);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
