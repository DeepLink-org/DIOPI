/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    BEGIN_CALL_ACL_OP(out, input, index);
    TORCH_CHECK(inputAt.scalar_type() == outAt.scalar_type(), "index select need same dtype.");
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }

    at::Tensor indexTempAt = indexAt;
    if (indexAt.scalar_type() != at::kInt || indexAt.scalar_type() != at::kLong) {
        indexTempAt = indexAt.to(at::kLong);
    }

    if (false) {
        acl_op::index_select_out(inputAt, dim, indexTempAt, outAt);
    } else {
        op_api::index_select_out(inputAt, dim, indexTempAt, outAt);
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t grad, diopiSize_t inputSizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    BEGIN_CALL_ACL_OP(gradInput, grad, index);
    TORCH_CHECK(gradAt.scalar_type() == gradInputAt.scalar_type(), "index select backward need same dtype.");
    if (dim < 0) {
        dim = dim + inputSizes.len;
    }

    at::Tensor indexTempAt = indexAt;
    if (indexAt.scalar_type() != at::kInt || indexAt.scalar_type() != at::kLong) {
        indexTempAt = indexAt.to(at::kLong);
    }

    at::Scalar zero{0.0};
    op_api::fill_(gradInputAt, zero);
    at::Scalar one{1.0};

    if (false) {
        acl_op::index_add_out(gradInputAt, dim, indexTempAt, gradAt, one, gradInputAt);
    } else {
        op_api::index_add_out(gradInputAt, dim, indexTempAt, gradAt, one, gradInputAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
