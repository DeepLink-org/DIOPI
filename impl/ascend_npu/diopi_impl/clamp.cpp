/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <half.hpp>

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {
diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    BEGIN_CALL_ACL_OP(out, input, min, max);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
    if (minAt.defined() && maxAt.defined()) {
        op_api::clamp_out(inputTmp, minAt, maxAt, outAt);
    } else {
        if (minAt.defined() && !maxAt.defined()) {
            op_api::clamp_min_out(inputTmp, minAt, outAt);
        }
        if (maxAt.defined() && !minAt.defined()) {
            op_api::clamp_max_out(inputTmp, maxAt, outAt);
        }
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    if (min != nullptr && max != nullptr) {
        BEGIN_CALL_ACL_OP(out, input, min, max);
        if (inputAt.numel() == 0) {
            return diopiSuccess;
        }
        at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
        op_api::clamp_out(inputTmp, minAt, maxAt, outAt);
        END_CALL_ACL_OP();
    } else {
        if (min != nullptr) {
            BEGIN_CALL_ACL_OP(out, input, min);
            if (inputAt.numel() == 0) {
                return diopiSuccess;
            }
            at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
            op_api::clamp_min_out(inputTmp, minAt, outAt);
            END_CALL_ACL_OP();
        }
        if (max != nullptr) {
            BEGIN_CALL_ACL_OP(out, input, max);
            if (inputAt.numel() == 0) {
                return diopiSuccess;
            }
            at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
            EXEC_NPU_CMD(aclnnClampMaxTensor, inputTmp, maxAt, outAt);
            END_CALL_ACL_OP();
        }
    }
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    BEGIN_CALL_ACL_OP(input, min, max);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    if (minAt.defined() && maxAt.defined()) {
        op_api::clamp_(inputAt, minAt, maxAt);
    } else {
        if (minAt.defined()) {
            op_api::clamp_min_(inputAt, minAt);
        }
        if (maxAt.defined()) {
            op_api::clamp_max_(inputAt, maxAt);
        }
    }
    END_CALL_ACL_OP();
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    if (min != nullptr && max != nullptr) {
        BEGIN_CALL_ACL_OP(input, min, max);
        if (inputAt.numel() == 0) {
            return diopiSuccess;
        }
        op_api::clamp_(inputAt, minAt, maxAt);
        END_CALL_ACL_OP();
    } else {
        if (min != nullptr) {
            BEGIN_CALL_ACL_OP(input, min);
            if (inputAt.numel() == 0) {
                return diopiSuccess;
            }
            op_api::clamp_min_(inputAt, minAt);
            END_CALL_ACL_OP();
        }
        if (max != nullptr) {
            BEGIN_CALL_ACL_OP(input, max);
            if (inputAt.numel() == 0) {
                return diopiSuccess;
            }
            op_api::clamp_max_(inputAt, maxAt);
            END_CALL_ACL_OP();
        }
    }
}

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    BEGIN_CALL_ACL_OP(input, min);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::clamp_min_(inputAt, minAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    BEGIN_CALL_ACL_OP(input, min);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::clamp_min_(inputAt, minAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    BEGIN_CALL_ACL_OP(out, input, min);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
    op_api::clamp_min_out(inputTmp, minAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    BEGIN_CALL_ACL_OP(out, input, min);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
    op_api::clamp_min_out(inputTmp, minAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    BEGIN_CALL_ACL_OP(input, max);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::clamp_max_(inputAt, maxAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    BEGIN_CALL_ACL_OP(input, max);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::clamp_max_(inputAt, maxAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    BEGIN_CALL_ACL_OP(out, input, max);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
    EXEC_NPU_CMD(aclnnClampMaxTensor, inputTmp, maxAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    BEGIN_CALL_ACL_OP(out, input, max);
    if (inputAt.numel() == 0) {
        return diopiSuccess;
    }
    at::Tensor inputTmp = inputAt.to(outAt.scalar_type(), true);
    op_api::clamp_max_out(inputTmp, maxAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
