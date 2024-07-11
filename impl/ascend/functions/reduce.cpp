/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    int64_t numel;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, out);
        return diopiSuccess;
    }

    diopiSize_t inputSize, outSize;
    diopiGetTensorShape(input, &inputSize);
    if (0 == inputSize.len) {
        return diopiCopyInp(ctx, input, out);
    }
    diopiGetTensorShape(out, &outSize);
    bool keepDim = true;
    if (inputSize.len != outSize.len) {
        keepDim = false;
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    aclDataType type = diopiDtypeToAclDataType(dtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnReduceSum, ctx, input, dim, keepDim, type, out);
    return diopiSuccess;
}

diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiSize_t inputSize, outSize;
    diopiGetTensorShape(input, &inputSize);
    diopiGetTensorShape(out, &outSize);
    bool keepDim = true;
    if (inputSize.len != outSize.len) {
        keepDim = false;
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    aclDataType type = diopiDtypeToAclDataType(dtype);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMean, ctx, input, dim, keepDim, type, out);
    return diopiSuccess;
}

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    bool keepdim = false;
    if (inputAt.dim() == outAt.dim()) {
        keepdim = true;
    }

    int64_t correction = 0;
    if (unbiased) {
        correction = 1;
    }

    if (dim.data == nullptr || dim.len == 0) {
        std::vector<int64_t> allDim(inputAt.dim());
        std::iota(allDim.begin(), allDim.end(), 0);
        diopiSize_t rDim = vectorToDiopiSize(allDim);
        DIOPI_ASCEND_CALL_ACLNN(aclnnStd, ctx, input, rDim, correction, keepdim, out);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnStd, ctx, input, dim, correction, keepdim, out);
    }
    return diopiSuccess;
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiSize_t inputSize, outSize;
    diopiGetTensorShape(input, &inputSize);
    diopiGetTensorShape(out, &outSize);
    bool keepDim = true;
    if (inputSize.len != outSize.len) {
        keepDim = false;
    }

    if (nullptr == dim) {
        std::vector<int64_t> dimVector(inputSize.len);
        std::iota(dimVector.begin(), dimVector.end(), 0);
        diopiSize_t dimSize = vectorToDiopiSize(dimVector);
        DIOPI_ASCEND_CALL_ACLNN(aclnnAll, ctx, input, dimSize, keepDim, out);
    } else {
        std::array<int64_t, 1> dimVector = {*dim};
        diopiSize_t dimSize = {dimVector.data(), dimVector.size()};
        DIOPI_ASCEND_CALL_ACLNN(aclnnAll, ctx, input, dimSize, keepDim, out);
    }
    return diopiSuccess;
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiSize_t inputSize, outSize;
    diopiGetTensorShape(input, &inputSize);
    diopiGetTensorShape(out, &outSize);
    bool keepDim = true;
    if (inputSize.len != outSize.len) {
        keepDim = false;
    }

    if (nullptr == dim) {
        std::vector<int64_t> dimVector(inputSize.len);
        std::iota(dimVector.begin(), dimVector.end(), 0);
        diopiSize_t dimSize = vectorToDiopiSize(dimVector);
        DIOPI_ASCEND_CALL_ACLNN(aclnnAny, ctx, input, dimSize, keepDim, out);
    } else {
        std::array<int64_t, 1> dimVector = {*dim};
        diopiSize_t dimSize = {dimVector.data(), dimVector.size()};
        DIOPI_ASCEND_CALL_ACLNN(aclnnAny, ctx, input, dimSize, keepDim, out);
    }
    return diopiSuccess;
}

diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    aclDataType type = diopiDtypeToAclDataType(dtype);

    if (nullptr == dim) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnProd, ctx, input, type, out);
    } else {
        diopiSize_t inputSize, outSize;
        diopiGetTensorShape(input, &inputSize);
        diopiGetTensorShape(out, &outSize);
        bool keepDim = true;
        if (inputSize.len != outSize.len) {
            keepDim = false;
        }
        DIOPI_ASCEND_CALL_ACLNN(aclnnProdDim, ctx, input, *dim, keepDim, type, out);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
