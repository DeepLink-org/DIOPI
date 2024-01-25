/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "common.hpp"

namespace impl {
namespace camb {
diopiError_t broadcastContiguous(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input) {
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
    DIOPI_CALL_CNNL(cnnlExpand(handle, inputDesc.get(), const_cast<DiopiTensor&>(input).data(), outDesc.get(), out.data()));
    return diopiSuccess;
}

diopiError_t broadcastContiguous(diopiContextHandle_t ctx, DiopiTensor inputTensor, const std::vector<int64_t>& targetShape, diopiDtype_t targetDtype,
                                 DiopiTensor* outTensor) {
    DiopiTensor bcastInputTensor;
    if (inputTensor.shape() != targetShape) {
        bcastInputTensor = requiresTensor(ctx, vec2diopiSizeT(targetShape), targetDtype);
        DIOPI_CALL(broadcastContiguous(ctx, bcastInputTensor, inputTensor));
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
    std::vector<int64_t> srcStridesAfterBroadCast(targetNDims, 0);
    if (srcNDims > targetNDims) {
        return false;
    }
    for (int64_t iSrcDim = srcNDims - 1, iTargetDim = targetNDims - 1; iSrcDim >= 0; iSrcDim--, iTargetDim--) {
        if (srcShape[iSrcDim] == targetShape[iTargetDim]) {
            srcStridesAfterBroadCast[iTargetDim] = srcStride[iSrcDim];
        } else if (srcShape[iSrcDim] == 1) {
            srcStridesAfterBroadCast[iTargetDim] = 0;
        } else {
            return false;
        }
    }
    outStrides = std::move(srcStridesAfterBroadCast);
    return true;
}

bool broadcast(DiopiTensor inputTensor, const std::vector<int64_t>& targetShape, DiopiTensor* outTensor) {
    std::vector<int64_t> strides;
    if (checkBroadCast(inputTensor, targetShape, strides)) {
        *outTensor = inputTensor.asStrided(targetShape, strides);
        return true;
    }
    return false;
}

int isBroadcast(DiopiTensor& inputTensor, DiopiTensor& otherTensor) {
    // 0:cannot broadcast,1:inputTensor broadcasts,2:otherTensor broadcasts,3:both broadcast
    int dimA = inputTensor.dim();
    int dimB = otherTensor.dim();
    int minDim = dimA;
    int broadCastA = 0;
    int broadCastB = 0;

    if (dimA == 0) {
        broadCastA = 1;
    }

    if (dimB == 0) {
        broadCastB = 2;
    }

    if (dimA > dimB) {
        minDim = dimB;
    }

    for (int i = 1; i <= minDim; i++) {
        if (inputTensor.shape()[dimA - i] == otherTensor.shape()[dimB - i]) {
            continue;
        } else if (inputTensor.shape()[dimA - i] == 1) {
            broadCastA = 1;
            continue;
        } else if (otherTensor.shape()[dimB - i] == 1) {
            broadCastB = 2;
            continue;
        } else {
            return 0;
        }
    }
    return broadCastA + broadCastB;
}

}  // namespace camb
}  // namespace impl
