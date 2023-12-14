/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    if (diopi_dtype_bool == dtype) {
        AclOpRunner<1, 1>("LogicalNot", ctx).addInput(input).addOutput(out).run();
    } else {
        AclOpRunner<1, 1>("Invert", ctx).addInput(input).addOutput(out).run();
    }
    return diopiSuccess;
}

diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiBitwiseNot(ctx, input, input); }

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    AscendTensor inputTensor(input);
    AscendTensor otherTensor(other);

    if (diopi_dtype_bool == inputTensor.dtype()) {
        if (inputTensor.dtype() == otherTensor.dtype()) {
            AclOpRunner<2, 1>("LogicalAnd", ctx).addInput(input).addInput(other).addOutput(out).run();
        } else {
            diopiTensorHandle_t inputCopy;
            makeTensorLike(ctx, &inputCopy, input, otherTensor.dtype());
            diopiCastDtype(ctx, inputCopy, input);
            AclOpRunner<2, 1>("BitwiseAnd", ctx).addInput(inputCopy).addInput(other).addOutput(out).run();
        }
    } else {
        if (inputTensor.dtype() == otherTensor.dtype()) {
            AclOpRunner<2, 1>("BitwiseAnd", ctx).addInput(input).addInput(other).addOutput(out).run();
        } else {
            diopiTensorHandle_t otherCopy;
            makeTensorLike(ctx, &otherCopy, other, inputTensor.dtype());
            diopiCastDtype(ctx, otherCopy, other);
            AclOpRunner<2, 1>("BitwiseAnd", ctx).addInput(input).addInput(otherCopy).addOutput(out).run();
        }
    }

    return diopiSuccess;
}

diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return diopiBitwiseAnd(ctx, input, input, other);
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    AscendTensor inputTensor(input);
    if (diopi_dtype_bool == inputTensor.dtype()) {
        if (inputTensor.dtype() == other->stype) {
            AclOpRunner<2, 1>("LogicalAnd", ctx).addInput(input).addConstInput(*other, other->stype).addOutput(out).run();
        } else {
            diopiTensorHandle_t inputCopy;
            makeTensorLike(ctx, &inputCopy, input, other->stype);
            diopiCastDtype(ctx, inputCopy, input);
            AclOpRunner<2, 1>("BitwiseAnd", ctx).addInput(inputCopy).addConstInput(*other, other->stype).addOutput(out).run();
        }
    } else {
        AclOpRunner<2, 1>("BitwiseAnd", ctx).addInput(input).addConstInput(*other, inputTensor.dtype()).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return diopiBitwiseAndScalar(ctx, input, input, other);
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    AscendTensor inputTensor(input);
    AscendTensor otherTensor(other);

    if (diopi_dtype_bool == inputTensor.dtype()) {
        if (inputTensor.dtype() == otherTensor.dtype()) {
            AclOpRunner<2, 1>("LogicalOr", ctx).addInput(input).addInput(other).addOutput(out).run();
        } else {
            diopiTensorHandle_t inputCopy;
            makeTensorLike(ctx, &inputCopy, input, otherTensor.dtype());
            diopiCastDtype(ctx, inputCopy, input);
            AclOpRunner<2, 1>("BitwiseOr", ctx).addInput(inputCopy).addInput(other).addOutput(out).run();
        }
    } else {
        if (inputTensor.dtype() == otherTensor.dtype()) {
            AclOpRunner<2, 1>("BitwiseOr", ctx).addInput(input).addInput(other).addOutput(out).run();
        } else {
            diopiTensorHandle_t otherCopy;
            makeTensorLike(ctx, &otherCopy, other, inputTensor.dtype());
            diopiCastDtype(ctx, otherCopy, other);
            AclOpRunner<2, 1>("BitwiseOr", ctx).addInput(input).addInput(otherCopy).addOutput(out).run();
        }
    }

    return diopiSuccess;
}

diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return diopiBitwiseOr(ctx, input, input, other);
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    AscendTensor inputTensor(input);
    if (diopi_dtype_bool == inputTensor.dtype()) {
        if (inputTensor.dtype() == other->stype) {
            AclOpRunner<2, 1>("LogicalOr", ctx).addInput(input).addConstInput(*other, other->stype).addOutput(out).run();
        } else {
            diopiTensorHandle_t inputCopy;
            makeTensorLike(ctx, &inputCopy, input, other->stype);
            diopiCastDtype(ctx, inputCopy, input);
            AclOpRunner<2, 1>("BitwiseOr", ctx).addInput(inputCopy).addConstInput(*other, other->stype).addOutput(out).run();
        }
    } else {
        AclOpRunner<2, 1>("BitwiseOr", ctx).addInput(input).addConstInput(*other, inputTensor.dtype()).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return diopiBitwiseOrScalar(ctx, input, input, other);
}

}  // namespace ascend
}  // namespace impl
