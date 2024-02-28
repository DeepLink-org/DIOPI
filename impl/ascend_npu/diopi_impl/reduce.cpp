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
        return diopiSuccess;
    }

    if (inputAt.dim() == 0) {
        diopiCopyInp(ctx, input, out);
        return diopiSuccess;
    }

    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);

    if (inS.len != outS.len) {
        keepdim = false;
    }

    at::ArrayRef<int64_t> rdim(dim.data, dim.len);

    if (impl::aten::isIntegralTypeWithBool(impl::aten::getDIOPITensorType(inputAt.scalar_type()))) {
        op_api::sum_out(inputAt.to(at::kLong), rdim, keepdim, inputAt.scalar_type(), outAt);
    } else {
        op_api::sum_out(inputAt, rdim, keepdim, inputAt.scalar_type(), outAt);
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
