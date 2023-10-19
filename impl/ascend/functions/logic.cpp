/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t logic(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const char* logicOp) {
    diopiDtype_t inputDtype, otherDtype;
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    AclOpRunner<2, 1>(logicOp, ctx).addInput(input, highType).addInput(other, highType).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t logicInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const char* logicOp) {
    return logic(ctx, input, input, other, logicOp);
}

diopiError_t logicScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const char* logicOp) {
    diopiDtype_t inputDtype, otherDtype;
    diopiGetTensorDtype(input, &inputDtype);
    otherDtype = other->stype;
    diopiTensorHandle_t inputCopy, otherCopy;

    if (inputDtype != otherDtype && (!isIntegralType(otherDtype) && isIntegralType(inputDtype))) {
        inputCopy = contiguous(ctx, input, diopi_dtype_float32);
        makeTensorFromScalar(ctx, other, &otherCopy, diopi_dtype_float32);
    } else {
        inputCopy = contiguous(ctx, input);
        makeTensorFromScalar(ctx, other, &otherCopy, inputDtype);
    }
    AclOpRunner<2, 1>(logicOp, ctx).addInput(inputCopy).addConstInput(otherCopy).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t logicInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const char* logicOp) {
    return logicScalar(ctx, input, input, other, logicOp);
}

// ge
diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    return logicScalar(ctx, out, input, other, "GreaterEqual");
}

diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return logicInpScalar(ctx, input, other, "GreaterEqual");
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "GreaterEqual");
}

diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logicInp(ctx, input, other, "GreaterEqual");
}

// gt
diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    return logicScalar(ctx, out, input, other, "Greater");
}

diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return logicInpScalar(ctx, input, other, "Greater");
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "Greater");
}

diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) { return logicInp(ctx, input, other, "Greater"); }

// le
diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    return logicScalar(ctx, out, input, other, "LessEqual");
}

diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return logicInpScalar(ctx, input, other, "LessEqual");
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "LessEqual");
}

diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logicInp(ctx, input, other, "LessEqual");
}

// lt
diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    return logicScalar(ctx, out, input, other, "Less");
}

diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return logicInpScalar(ctx, input, other, "Less");
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "Less");
}

diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) { return logicInp(ctx, input, other, "Less"); }

// ne
diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    return logicScalar(ctx, out, input, other, "NotEqual");
}

diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return logicInpScalar(ctx, input, other, "NotEqual");
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "NotEqual");
}

diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) { return logicInp(ctx, input, other, "NotEqual"); }

// eq
diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    return logicScalar(ctx, out, input, other, "Equal");
}

diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return logicInpScalar(ctx, input, other, "Equal");
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "Equal");
}

diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) { return logicInp(ctx, input, other, "Equal"); }

//  logical_and
diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "LogicalAnd");
}

diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logicInp(ctx, input, other, "LogicalAnd");
}

// logical_or
diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logic(ctx, out, input, other, "LogicalOr");
}

diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return logicInp(ctx, input, other, "LogicalOr");
}

// logical_not
diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("LogicalNot", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiLogicalNot(ctx, input, input); }

}  // namespace ascend
}  // namespace impl
