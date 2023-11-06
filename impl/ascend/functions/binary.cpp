/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cmath>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

aclDataType dtypeConvertor(diopiDtype_t type) {
    auto dtype = getAclDataType(type);
    if (dtype == ACL_BOOL) {
        return ACL_UINT8;
    }
    return dtype;
}

bool isScalarOne(const diopiScalar_t* alpha) {
    if (alpha == nullptr) return true;
    if (alpha->stype == diopi_dtype_int64) {
        int val = getValue<int>(alpha);
        return val == 1;
    } else {
        float val = getValue<float>(alpha);
        return fabs(val - 1.0) < 1e-6;
    }
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    if (isScalarOne(alpha)) {
        AclOpRunner<2, 1, dtypeConvertor>("Add", ctx).addInput(input).addInput(other).addOutput(out).run();
    } else {
        diopiDtype_t outDtype, castType;
        diopiGetTensorDtype(out, &outDtype);

        if (isFloatingType(outDtype)) {
            castType = diopi_dtype_float32;
        } else {
            castType = diopi_dtype_int32;
        }

        AclOpRunner<3, 1>("AxpyV2", ctx).addInput(input, castType).addInput(other, castType).addConstInput(*alpha, diopi_dtype_float32).addOutput(out).run();
    }
    return diopiSuccess;
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    return diopiAdd(ctx, input, input, other, alpha);
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiAdd(ctx, out, input, trOther, alpha);
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    return diopiAddScalar(ctx, input, input, other, alpha);
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t outCopy;
    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }
    const float value = (alpha != nullptr) ? getValue<float>(alpha) : 1.0;
    if (value == 1.0) {
        AclOpRunner<2, 1, dtypeConvertor>("Sub", ctx).addInput(input, highType).addInput(other, highType).addOutput(outCopy).run();
    } else if (value == -1.0) {
        AclOpRunner<2, 1, dtypeConvertor>("AddV2", ctx).addInput(input, highType).addInput(other, highType).addOutput(outCopy).run();
    } else {
        if (diopi_dtype_float64 == highType) {
            highType = diopi_dtype_float32;
        }
        AclOpRunner<2, 1>("Axpy", ctx).addInput(input, highType).addInput(other, highType).setAttr<float>("alpha", -value).addOutput(outCopy).run();
    }
    if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
    return diopiSuccess;
}

diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    return diopiSub(ctx, input, input, other, alpha);
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiSub(ctx, out, input, trOther, alpha);
}

diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    return diopiSubScalar(ctx, input, input, other, alpha);
}

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t outCopy;
    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }
    AclOpRunner<2, 1, dtypeConvertor>("Mul", ctx).addInput(input, highType).addInput(other, highType).addOutput(outCopy).run();
    if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
    return diopiSuccess;
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) { return diopiMul(ctx, input, input, other); }

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t trOther = nullptr;
    makeTensorFromScalar(ctx, other, &trOther, diopi_dtype_float32, diopiDevice_t::diopi_device);
    return diopiMul(ctx, out, input, trOther);
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return diopiMulScalar(ctx, input, input, other);
}

diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      diopiRoundMode_t roundingMode) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t outCopy;

    // only in float data can we get nan or inf
    std::set<diopiDtype_t> supportDtype{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64, diopi_dtype_complex64, diopi_dtype_complex128};
    if (supportDtype.find(highType) == supportDtype.end()) {
        highType = diopi_dtype_float32;
    }

    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }

    if (RoundModeFloor == roundingMode) {
        // floor
        AclOpRunner<2, 1, dtypeConvertor>("FloorDiv", ctx).addInput(input, highType).addInput(other, highType).addOutput(outCopy).run();
        if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
        return diopiSuccess;
    } else {
        // default
        AclOpRunner<2, 1, dtypeConvertor>("RealDiv", ctx).addInput(input, highType).addInput(other, highType).addOutput(outCopy).run();
        if (RoundModeTrunc == roundingMode) {
            // trunc
            // trunc only support float16, float32, int8, uint8, int32.
            if (highType == diopi_dtype_float64) {
                AclOpRunner<1, 1, dtypeConvertor>("Trunc", ctx).addInput(out, diopi_dtype_float32).addOutput(out).run();
            } else {
                AclOpRunner<1, 1, dtypeConvertor>("Trunc", ctx).addInput(out, highType).addOutput(out).run();
            }
        }
        if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
        return diopiSuccess;
    }
}

diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    return diopiDiv(ctx, input, input, other, roundingMode);
}

diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            diopiRoundMode_t roundingMode) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiDiv(ctx, out, input, trOther, roundingMode);
}

diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t roundingMode) {
    return diopiDivScalar(ctx, input, input, other, roundingMode);
}

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    // as this op do not support BOOL, UIT8, and Int16, these three data types are converted to Int32
    if (dtype == diopi_dtype_bool || dtype == diopi_dtype_uint8 || dtype == diopi_dtype_int16) {
        AscendTensor inputCopy(input);
        AscendTensor otherCopy(other);
        castTensor(ctx, inputCopy, diopi_dtype_int32);
        castTensor(ctx, otherCopy, diopi_dtype_int32);
        AclOpRunner<2, 1>("Maximum", ctx).addInput(inputCopy).addInput(otherCopy).addOutput(out).run();
    } else {
        AclOpRunner<2, 1>("Maximum", ctx).addInput(input, dtype).addInput(other, dtype).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    AscendTensor inputAt(input), otherAt(other), outAt(out);
    if (0 == inputAt.numel() || 0 == otherAt.numel()) {
        diopiScalar_t zero = constructDiopiScalarT(inputAt.dtype(), 0);
        diopiFill(ctx, out, &zero);
        return diopiSuccess;
    }
    AclOpRunner<2, 1>("Minimum", ctx).addInput(input, outAt.dtype()).addInput(other, outAt.dtype()).addOutput(out).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
