/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

// ge
diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnGeScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceGeScalar, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnGeTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceGeTensor, ctx, input, other);
    return diopiSuccess;
}

// gt
diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnGtScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceGtScalar, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnGtTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceGtTensor, ctx, input, other);
    return diopiSuccess;
}

// le
diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLeScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLeScalar, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLeTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLeTensor, ctx, input, other);
    return diopiSuccess;
}

// lt
diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLtScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLtScalar, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLtTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLtTensor, ctx, input, other);
    return diopiSuccess;
}

// ne
diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnNeScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceNeScalar, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnNeTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceNeTensor, ctx, input, other);
    return diopiSuccess;
}

// eq
diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnEqScalar, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceEqScalar, ctx, input, other);
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnEqTensor, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceEqTensor, ctx, input, other);
    return diopiSuccess;
}

//  logical_and
diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLogicalAnd, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLogicalAnd, ctx, input, other);
    return diopiSuccess;
}

// logical_or
diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLogicalOr, ctx, input, other, out);
    return diopiSuccess;
}

diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLogicalOr, ctx, input, other);
    return diopiSuccess;
}

// logical_not
diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnLogicalNot, ctx, input, out);
    return diopiSuccess;
}

diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceLogicalNot, ctx, input);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
