/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

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

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t inputCopy, otherCopy, outCopy;
    if (inputDtype != highType) {
        makeTensorLike(ctx, &inputCopy, input, highType);
        diopiCastDtype(ctx, inputCopy, input);
    } else {
        inputCopy = const_cast<diopiTensorHandle_t>(input);
    }
    if (otherDtype != highType) {
        makeTensorLike(ctx, &otherCopy, other, highType);
        diopiCastDtype(ctx, otherCopy, other);
    } else {
        otherCopy = const_cast<diopiTensorHandle_t>(other);
    }
    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }
    const float value = (alpha != nullptr) ? getValue<float>(alpha) : 1.0;
    if (value == 1.0) {
        AclOpRunner<2, 1, dtypeConvertor>("AddV2", ctx).addInput(inputCopy, otherCopy).addOutput(outCopy).run();
    } else {
        AclOpRunner<2, 1>("Axpy", ctx).addInput(inputCopy).addInput(otherCopy).setAttr<float>("alpha", value).addOutput(outCopy).run();
    }
    if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    return diopiAdd(ctx, input, input, other, alpha);
}

DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiAdd(ctx, out, input, trOther, alpha);
}

DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    return diopiAddScalar(ctx, input, input, other, alpha);
}

DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t inputCopy, otherCopy, outCopy;
    if (inputDtype != highType) {
        makeTensorLike(ctx, &inputCopy, input, highType);
        diopiCastDtype(ctx, inputCopy, input);
    } else {
        inputCopy = const_cast<diopiTensorHandle_t>(input);
    }
    if (otherDtype != highType) {
        makeTensorLike(ctx, &otherCopy, other, highType);
        diopiCastDtype(ctx, otherCopy, other);
    } else {
        otherCopy = const_cast<diopiTensorHandle_t>(other);
    }
    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }
    const float value = (alpha != nullptr) ? getValue<float>(alpha) : 1.0;
    if (value == 1.0) {
        AclOpRunner<2, 1, dtypeConvertor>("Sub", ctx).addInput(inputCopy, otherCopy).addOutput(outCopy).run();
    } else if (value == -1.0) {
        AclOpRunner<2, 1, dtypeConvertor>("AddV2", ctx).addInput(inputCopy, otherCopy).addOutput(outCopy).run();
    } else {
        AclOpRunner<2, 1>("Axpy", ctx).addInput(inputCopy).addInput(otherCopy).setAttr<float>("alpha", -value).addOutput(outCopy).run();
    }
    if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    return diopiSub(ctx, input, input, other, alpha);
}

DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiSub(ctx, out, input, trOther, alpha);
}

DIOPI_API diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    return diopiSubScalar(ctx, input, input, other, alpha);
}

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t inputCopy, otherCopy, outCopy;
    if (inputDtype != highType) {
        makeTensorLike(ctx, &inputCopy, input, highType);
        diopiCastDtype(ctx, inputCopy, input);
    } else {
        inputCopy = const_cast<diopiTensorHandle_t>(input);
    }
    if (otherDtype != highType) {
        makeTensorLike(ctx, &otherCopy, other, highType);
        diopiCastDtype(ctx, otherCopy, other);
    } else {
        otherCopy = const_cast<diopiTensorHandle_t>(other);
    }
    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }
    AclOpRunner<2, 1, dtypeConvertor>("Mul", ctx).addInput(inputCopy, otherCopy).addOutput(outCopy).run();
    if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return diopiMul(ctx, input, input, other);
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t trOther = nullptr;
    makeTensorFromScalar(ctx, other, &trOther, diopi_dtype_float32, diopiDevice_t::diopi_device);
    return diopiMul(ctx, out, input, trOther);
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    return diopiMulScalar(ctx, input, input, other);
}

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t roundingMode) {
    diopiDtype_t outDtype, inputDtype, otherDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(other, &otherDtype);
    diopiDtype_t highType = promoteTypes(inputDtype, otherDtype);
    diopiTensorHandle_t inputCopy, otherCopy, outCopy;
    if (inputDtype != highType) {
        makeTensorLike(ctx, &inputCopy, input, highType);
        diopiCastDtype(ctx, inputCopy, input);
    } else {
        inputCopy = const_cast<diopiTensorHandle_t>(input);
    }
    if (otherDtype != highType) {
        makeTensorLike(ctx, &otherCopy, other, highType);
        diopiCastDtype(ctx, otherCopy, other);
    } else {
        otherCopy = const_cast<diopiTensorHandle_t>(other);
    }
    if (outDtype != highType) {
        makeTensorLike(ctx, &outCopy, out, highType);
    } else {
        outCopy = out;
    }
    AclOpRunner<2, 1, dtypeConvertor>("RealDiv", ctx).addInput(inputCopy, otherCopy).addOutput(outCopy).run();
    if (outDtype != highType) diopiCastDtype(ctx, out, outCopy);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    return diopiDiv(ctx, input, input, other, roundingMode);
}

DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      diopiRoundMode_t roundingMode) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    makeTensorFromScalar(ctx, other, &trOther, dtype, diopiDevice_t::diopi_device);
    return diopiDiv(ctx, out, input, trOther, roundingMode);
}

DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t roundingMode) {
    return diopiDivScalar(ctx, input, input, other, roundingMode);
}

}  // namespace ascend
}  // namespace impl
