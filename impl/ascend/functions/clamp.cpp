/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampTensor, ctx, input, min, max, out);
    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minPtr,
                              const diopiScalar_t* maxPtr) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    if (minPtr != nullptr && maxPtr != nullptr) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnClamp, ctx, input, minPtr, maxPtr, out);
    } else {
        if (minPtr != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, input, minPtr, out);
        }
        if (maxPtr != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMax, ctx, input, maxPtr, out);
        }
    }

    return diopiSuccess;
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampTensor, ctx, input, min, max, input);
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnClamp, ctx, input, min, max, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, input, min, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMinTensor, ctx, input, min, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, input, min, out);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampMinTensor, ctx, input, min, out);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMax, ctx, input, max);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMaxTensor, ctx, input, max);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMax, ctx, input, max, out);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (input == nullptr || inputNumel == 0) {
        return diopiSuccess;
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMaxTensor, ctx, input, max, out);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
