/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

// ::std::tuple<at::Tensor &,at::Tensor &> min_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices);
diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t minIndices, diopiConstTensorHandle_t input, int64_t dim) {
    BEGIN_CALL_ACL_OP(input, min, minIndices);
    if (input == nullptr || inputAt.numel() <= 0) {
        return diopiSuccess;
    }
    bool keepdim = minAt.dim() == inputAt.dim();
    op_api::min_out(inputAt, dim, keepdim, minAt, minIndicesAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
