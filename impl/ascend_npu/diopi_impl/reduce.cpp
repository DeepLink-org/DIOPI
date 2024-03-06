/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    BEGIN_CALL_ACL_OP(input, out);
    if (inputAt.numel() == 0) {
        op_api::fill_(outAt, c10::Scalar(0.0));
        END_CALL_ACL_OP();
    }

    if (inputAt.dim() == 0) {
        diopiCopyInp(ctx, input, out);
        END_CALL_ACL_OP();
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
        END_CALL_ACL_OP();
    }

    if (inputAt.dim() == 0) {
        diopiCopyInp(ctx, input, out);
        END_CALL_ACL_OP();
    }

    bool keepdim = true;
    if (inputAt.dim() != outAt.dim()) {
        keepdim = false;
    }

    at::ArrayRef<int64_t> rdim(dim.data, dim.len);

    op_api::mean_out(inputAt, rdim, keepdim, outAt.scalar_type(), outAt);

    END_CALL_ACL_OP();
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    BEGIN_CALL_ACL_OP(input, out);

    if (inputAt.numel() == 0) {
        op_api::fill_(outAt, c10::Scalar(0.0));
        END_CALL_ACL_OP();
    }

    if (dim == nullptr) {
        op_api::any_out(inputAt, outAt);
    } else {
        bool keepdim = false;
        if (inputAt.dim() == outAt.dim()) {
            keepdim = true;
        }
        op_api::any_out(inputAt, *dim, keepdim, outAt);
    }

    END_CALL_ACL_OP();
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    BEGIN_CALL_ACL_OP(input, out);
    at::IntArrayRef dims;
    if (dim) {
        dims = at::IntArrayRef(dim, 1);
    }

    bool keepDim = false;
    EXEC_NPU_CMD(aclnnAll, inputAt, dims, keepDim, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    BEGIN_CALL_ACL_OP(input, out);
    bool keepDim = false;
    auto dtype = outAt.scalar_type();
    EXEC_NPU_CMD(aclnnProdDim, inputAt, *dim, keepDim, dtype, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
