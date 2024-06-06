/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

static int64_t getReduce(const char* reduce) {
    if (strcmp(reduce, "add") == 0) {
        return 1;
    } else if (strcmp(reduce, "multiply") == 0) {
        return 2;
    } else {
        return 0;
    }
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                          diopiConstTensorHandle_t index, const char* reduce) {
    BEGIN_CALL_ACL_OP(input, src, index, out);
    if (inputAt.numel() <= 0 || !inputAt.defined() || inputAt.dim() == 0) {
        return diopiSuccess;
    }
    // check index
    TORCH_CHECK((inputAt.dim() == srcAt.dim() && inputAt.dim() == indexAt.dim()) || indexAt.dim() == 0,
                "input,src,index must have same ndim! only exception is index is empty");
    if (indexAt.dim() == 0) {
        outAt.copy_(inputAt, true);
        return diopiSuccess;
    }
    // input to output type
    at::Tensor inputTmpAt = inputAt;
    if (outAt.scalar_type() != inputAt.scalar_type()) {
        inputTmpAt = inputAt.to(outAt.scalar_type(), true);
    }
    int64_t reduction = getReduce(reduce);
    EXEC_NPU_CMD(aclnnScatter, inputTmpAt, dim, indexAt, srcAt, reduction, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index,
                             const char* reduce) {
    BEGIN_CALL_ACL_OP(input, src, index);
    if (inputAt.numel() <= 0 || !inputAt.defined() || inputAt.dim() == 0) {
        return diopiSuccess;
    }
    // check index
    TORCH_CHECK((inputAt.dim() == srcAt.dim() && inputAt.dim() == indexAt.dim()) || indexAt.dim() == 0,
                "input,src,index must have same ndim! only exception is index is empty");
    if (indexAt.dim() == 0) {
        return diopiSuccess;
    }
    int64_t reduction = getReduce(reduce);
    EXEC_NPU_CMD(aclnnInplaceScatter, inputAt, dim, indexAt, srcAt, reduction);
    END_CALL_ACL_OP();
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                diopiConstTensorHandle_t index, const char* reduce) {
    // TODO(shenhao): replace aclnnInplaceScatterValue with aclnnScatterValue after fixing aclnnScatterValue for many wrong zeros;
    // BEGIN_CALL_ACL_OP(input, index, value, out);
    // if (inputAt.numel() <= 0 || !inputAt.defined() || inputAt.dim() == 0) {
    //     return diopiSuccess;
    // }
    // // input to output type
    // at::Tensor inputTmpAt = inputAt;
    // if (outAt.scalar_type() != inputAt.scalar_type()) {
    //     inputTmpAt = inputAt.to(outAt.scalar_type());
    // }
    // // check ndex
    // TORCH_CHECK(inputAt.dim() == indexAt.dim() || indexAt.dim() == 0, "input,index must have same ndim! only exception is index is empty");
    // if (indexAt.dim() == 0) {
    //     outAt.copy_(inputAt);
    //     return diopiSuccess;
    // }
    // int64_t reduction = getReduce(reduce);
    // EXEC_NPU_CMD(aclnnScatterValue, inputTmpAt, dim, indexAt, valueAt, reduction, outAt);
    // END_CALL_ACL_OP();
    BEGIN_CALL_ACL_OP(input, index, value, out);
    if (inputAt.numel() <= 0 || !inputAt.defined() || inputAt.dim() == 0) {
        return diopiSuccess;
    }
    // check index
    TORCH_CHECK(inputAt.dim() == indexAt.dim() || indexAt.dim() == 0, "input,index must have same ndim! only exception is index is empty");
    outAt.copy_(inputAt, true);
    if (indexAt.dim() == 0) {
        return diopiSuccess;
    }
    int64_t reduction = getReduce(reduce);
    EXEC_NPU_CMD(aclnnInplaceScatterValue, outAt, dim, indexAt, valueAt, reduction);
    END_CALL_ACL_OP();
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index,
                                   const char* reduce) {
    BEGIN_CALL_ACL_OP(input, value, index);
    if (inputAt.numel() <= 0 || !inputAt.defined() || inputAt.dim() == 0) {
        return diopiSuccess;
    }
    // check and prepare index
    at::Tensor indexTmpAt = indexAt;
    TORCH_CHECK(inputAt.dim() == indexAt.dim() || indexAt.dim() == 0, "input,index must have same ndim! only exception is index is empty");
    if (indexAt.dim() == 0) {
        return diopiSuccess;
    }
    int64_t reduction = getReduce(reduce);
    EXEC_NPU_CMD(aclnnInplaceScatterValue, inputAt, dim, indexTmpAt, valueAt, reduction);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
