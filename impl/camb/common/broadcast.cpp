/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "common.hpp"

namespace impl {
namespace camb {
diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // check whether input.shape() match the targetShape
    std::vector<int64_t> targetShape = out.shape();
    std::vector<int64_t> inputShape = input.shape();
    int64_t nDimsTarget = targetShape.size();
    int64_t nDimsInput = inputShape.size();
    if (nDimsInput < nDimsTarget) {
        inputShape.insert(inputShape.begin(), nDimsTarget - nDimsInput, 1);
    }

    for (int i = 0; i < nDimsTarget; i++) {
        DIOPI_CHECK(((inputShape[i] == 1) || (inputShape[i] == targetShape[i])), "shape1 not match shape2, can't broadcast");
    }
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), const_cast<DiopiTensor&>(input).data(), outDesc.get(), out.data()));
    return diopiSuccess;
}

diopiError_t broadcastHelper(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor targetTensor, DiopiTensor* outTensor) {
    DiopiTensor bcastInputTensor;
    if (inputTensor.shape() != targetTensor.shape()) {
        bcastInputTensor = requiresTensor(ctx, vec2diopiSizeT(targetTensor.shape()), targetTensor.dtype());
        DIOPI_CALL(broadcast(ctx, bcastInputTensor, inputTensor));
    } else {
        bcastInputTensor = inputTensor;
    }
    *outTensor = bcastInputTensor;
    return diopiSuccess;
}

bool checkBroadCast(const DiopiTensor& src, const std::vector<int64_t>& targetShape, std::vector<int64_t>& outStrides) {
    std::vector<int64_t> srcShape = src.shape();
    std::vector<int64_t> srcStride = src.stride();
    int64_t srcNDims = src.dim();
    int64_t targetNDims = targetShape.size();
    if (srcNDims > targetNDims) {
        return false;
    }
    for (int64_t iSrcDim = srcNDims - 1, iTargetDim = targetNDims - 1; iSrcDim >= 0; iSrcDim--, iTargetDim--) {
        if (srcShape[iSrcDim] == targetShape[iTargetDim]) {
            outStrides[iTargetDim] = srcStride[iSrcDim];
        } else if (srcShape[iSrcDim] == 1) {
            outStrides[iSrcDim] = 0;
        } else {
            return false;
        }
    }
    return true;
}

bool broadcast1(DiopiTensor inputTensor, const std::vector<int64_t>& targetShape, DiopiTensor* outTensor) {
    std::vector<int64_t> strides;
    if (checkBroadCast(inputTensor, targetShape, strides)) {
        *outTensor = inputTensor.asStrided(targetShape, strides);
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}
}  // namespace camb
}  // namespace impl
