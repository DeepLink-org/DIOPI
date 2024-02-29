/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    BEGIN_CALL_ACL_OP(input, out);
    if (inputAt.numel() == 0) {
        op_api::fill_(outAt, c10::Scalar(0.0));
        return diopiSuccess;
    }

    if (inputAt.dim() == 0) {
        diopiCopyInp(ctx, input, out);
        return diopiSuccess;
    }

    bool keepdim = true;
    if (inputAt.dim() != outAt.dim()) {
        keepdim = false;
    }

    at::ArrayRef<int64_t> rdim(dim.data, dim.len);

    op_api::sum_out(inputAt, rdim, keepdim, outAt.scalar_type(), outAt);

    END_CALL_ACL_OP();
}

diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    BEGIN_CALL_ACL_OP(input, out);
    if (inputAt.numel() == 0) {
        op_api::fill_(outAt, c10::Scalar(std::nanf("")));
        return diopiSuccess;
    }

    if (inputAt.dim() == 0) {
        diopiCopyInp(ctx, input, out);
        return diopiSuccess;
    }

    bool keepdim = true;
    if (inputAt.dim() != outAt.dim()) {
        keepdim = false;
    }

    at::ArrayRef<int64_t> rdim(dim.data, dim.len);

    op_api::mean_out(inputAt, rdim, keepdim, outAt.scalar_type(), outAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
