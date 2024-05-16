/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);
    if (0 == inputShape.len) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, out, input);
        return diopiSuccess;
    }

    if (dim0 < 0) dim0 = dim0 + inputShape.len;
    if (dim1 < 0) dim1 = dim1 + inputShape.len;
    std::vector<int64_t> dims(inputShape.len);
    std::iota(dims.begin(), dims.end(), 0);
    dims[dim0] = dim1;
    dims[dim1] = dim0;
    diopiSize_t permuteDims = vectorToDiopiSize(dims);
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, input, permuteDims, out);
    return diopiSuccess;
}

diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    AscendTensor inputTensor(input);
    if (inputTensor.dim() == 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, out, input);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, input, dims, out);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
