/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    AscendTensor inputAt(input);

    int64_t dimTmp;
    if (dim == nullptr) {
        dimTmp = 0;
        std::vector<int64_t> flattenShape{inputAt.numel()};
        auto flattenInput = inputAt.view(flattenShape);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArgMax, ctx, flattenInput, dimTmp, keepdim, out);

    } else {
        dimTmp = *dim;
        DIOPI_ASCEND_CALL_ACLNN(aclnnArgMax, ctx, input, dimTmp, keepdim, out);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
