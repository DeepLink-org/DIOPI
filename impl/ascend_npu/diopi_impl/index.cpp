/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/DIOPIAdapter.h"

namespace impl::aten {
c10::List<c10::optional<at::Tensor>> castIntIndicesToLongIndices(const c10::List<c10::optional<at::Tensor>>& indices);
};

namespace OP_IMPL_NS {

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    BEGIN_CALL_ACL_OP(input);
    torch::List<c10::optional<at::Tensor>> indicesAtList;
    indicesAtList.reserve(nums);
    for (int i = 0; i < nums; ++i) {
        indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    }
    auto indicesCast = impl::aten::castIntIndicesToLongIndices(indicesAtList);

    at::Tensor outAt = op_api::index(inputAt, indicesCast);
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

    op_api::index_put_(zerosLikeInputAt, indicesAtList, gradOutputAt, true);
    gradInputAt.copy_(zerosLikeInputAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
