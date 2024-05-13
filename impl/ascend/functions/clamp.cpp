/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    castTensor(ctx, inputAt, outAt.dtype());

    if (min != nullptr && max != nullptr) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnClampTensor, ctx, inputAt, min, max, outAt);
    } else {
        if (max != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMaxTensor, ctx, inputAt, max, outAt);
        } else {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMinTensor, ctx, inputAt, min, outAt);
        }
    }

    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minPtr,
                              const diopiScalar_t* maxPtr) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    castTensor(ctx, inputAt, outAt.dtype());

    if (minPtr != nullptr && maxPtr != nullptr) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnClamp, ctx, inputAt, minPtr, maxPtr, outAt);
    } else {
        if (minPtr != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, inputAt, minPtr, outAt);
        }
        if (maxPtr != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMax, ctx, inputAt, maxPtr, outAt);
        }
    }

    return diopiSuccess;
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    if (min != nullptr && max != nullptr) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnClampTensor, ctx, input, min, max, input);
    } else {
        if (max != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMaxTensor, ctx, input, max);
        } else {
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMinTensor, ctx, input, min);
        }
    }

    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    if (min != nullptr && max != nullptr) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnClamp, ctx, input, min, max, input);
    } else {
        if (max != nullptr) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMax, ctx, input, max, input);
        } else {
            DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, input, min, input);
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, input, min, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMinTensor, ctx, input, min, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    castTensor(ctx, inputAt, outAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampMin, ctx, inputAt, min, outAt);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    castTensor(ctx, inputAt, outAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnClampMinTensor, ctx, inputAt, min, outAt);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMax, ctx, input, max);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMaxTensor, ctx, input, max);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    AscendTensor inputAt(input);
    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMax, ctx, input, max, out);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    AscendTensor inputAt(input);
    AscendTensor outAt(out);

    if (input == nullptr || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    castTensor(ctx, inputAt, outAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceClampMaxTensor, ctx, inputAt, max, outAt);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
