/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                       bool descending, const bool* pStable) {
    bool stable = (pStable == nullptr) ? false : *pStable;
    DIOPI_ASCEND_CALL_ACLNN(aclnnSort, ctx, input, stable, dim, descending, values, indices);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
