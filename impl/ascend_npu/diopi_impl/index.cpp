/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    BEGIN_CALL_ACL_OP(input);
    torch::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(nums);
    for (int i = 0; i < nums; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    auto outAt = acl_op::index(inputAt, indicesAtList);
    impl::aten::buildDiopiTensor(ctx, outAt, out);
    END_CALL_ACL_OP();
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t zerosLikeInput, diopiConstTensorHandle_t* indices,
                                int64_t nums, diopiConstTensorHandle_t gradOutput) {
    BEGIN_CALL_ACL_OP(gradInput, zerosLikeInput, gradOutput);
    torch::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(nums);
    for (int i = 0; i < nums; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    gradInputAt.copy_(zerosLikeInputAt);
    acl_op::_index_put_impl_(gradInputAt, indicesAtList, gradOutputAt, true, false);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
