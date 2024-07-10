/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    diopiCopyInp(ctx, input, out);
    std::vector<diopiConstTensorHandle_t> indicesVec(indices, indices + indicesCounts);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, out, indicesVec, values, accumulate, false);
    return diopiSuccess;
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    std::vector<diopiConstTensorHandle_t> indicesVec(indices, indices + indicesCounts);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, input, indicesVec, values, accumulate, false);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
