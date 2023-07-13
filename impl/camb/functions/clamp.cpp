#include <diopi/functions.h>
#include <iostream>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {
extern "C" {

diopiError_t clampScalarCheck(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    diopiDtype_t inDtype, outDtype;
    diopiGetTensorDtype(input, &inDtype);
    diopiGetTensorDtype(out, &outDtype);
    auto boundPtr = min ? min : (max ? max : nullptr);
    DIOPI_CHECK(outDtype == inDtype || (nullptr != boundPtr && boundPtr->stype == diopi_dtype_float64 && outDtype == diopi_dtype_float32),
                "the dtype of output must be the same as input or bound");
    return diopiSuccess;
}

diopiError_t clampTensorCheck(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out) {
    diopiDtype_t inDtype, outDtype;
    diopiGetTensorDtype(input, &inDtype);
    diopiGetTensorDtype(out, &outDtype);
    DIOPI_CHECK(inDtype == outDtype, "the dtype of input and output must be the same")
    return diopiSuccess;
}

diopiError_t getClampBoundPtr(diopiContextHandle_t ctx, diopiConstTensorHandle_t bound, diopiDtype_t desireDtype, void** out) {
    if (nullptr != bound) {
        DiopiTensor boundTensor(bound);
        DIOPI_CHECK(boundTensor.numel() == 1, "only supported when min and max are scalar || one element Tensor currently");
        if ((!DiopiDataType::isInteger(desireDtype) || diopi_dtype_float32 != boundTensor.dtype()) && desireDtype != boundTensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, boundTensor, desireDtype));
        }
        *out = boundTensor.data();
        return diopiSuccess;
    }
    *out = nullptr;
    return diopiSuccess;
}

diopiError_t clampCommon(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out, diopiConstTensorHandle_t min,
                         diopiConstTensorHandle_t max) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    DiopiTensor output32Tensor = outputTensor;
    bool isFloat = false;
    if (min) {
        diopiDtype_t dtype;
        diopiGetTensorDtype(min, &dtype);
        isFloat = diopi_dtype_float32 == dtype;
    } else if (max) {
        diopiDtype_t dtype;
        diopiGetTensorDtype(max, &dtype);
        isFloat = diopi_dtype_float32 == dtype;
    }
    if (DiopiDataType::isInteger(inputTensor.dtype())) {
        if (!isFloat) {
            DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_int32));
            DIOPI_CALL(dataTypeCast(ctx, output32Tensor, diopi_dtype_int32));
        } else {
            DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, output32Tensor, diopi_dtype_float32));
        }
    } else if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, output32Tensor, diopi_dtype_float32));
    }

    void* minPtr = nullptr;
    void* maxPtr = nullptr;
    DIOPI_CALL(getClampBoundPtr(ctx, min, inputTensor.dtype(), &minPtr));
    DIOPI_CALL(getClampBoundPtr(ctx, max, inputTensor.dtype(), &maxPtr));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output32Desc(output32Tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(
        cnnlClip_v2(handle, CNNL_POINTER_MODE_DEVICE, inputDesc.get(), inputTensor.data(), minPtr, maxPtr, output32Desc.get(), output32Tensor.data()));
    if (outputTensor.dtype() != output32Tensor.dtype()) {
        if (outputTensor.dtype() != diopi_dtype_uint8) {
            DIOPI_CALL(dataTypeCast(ctx, outputTensor, output32Tensor));

        } else {
            DIOPI_CALL(dataTypeCast(ctx, output32Tensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, outputTensor, output32Tensor));
        }
    }
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    DIOPI_CHECK(min != nullptr || max != nullptr, "At least one of \'min\' or \'max\' must not be None");
    if (min == nullptr) {
        return diopiClampMaxInpScalar(ctx, input, max);
    } else if (max == nullptr) {
        return diopiClampMinInpScalar(ctx, input, min);
    }
    DiopiTensor minTensorTmp;
    DiopiTensor maxTensorTmp;
    makeTensorFromScalar(ctx, min, minTensorTmp);
    makeTensorFromScalar(ctx, max, maxTensorTmp);
    diopiTensorHandle_t minTensor = minTensorTmp.tensorHandle();
    diopiTensorHandle_t maxTensor = maxTensorTmp.tensorHandle();
    return clampCommon(ctx, input, input, minTensor, maxTensor);
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    DIOPI_CHECK(min != nullptr || max != nullptr, "At least one of \'min\' or \'max\' must not be None");
    if (min == nullptr) {
        return diopiClampMaxInp(ctx, input, max);
    } else if (max == nullptr) {
        return diopiClampMinInp(ctx, input, min);
    }
    return clampCommon(ctx, input, input, min, max);
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    DIOPI_CHECK(min != nullptr || max != nullptr, "At least one of \'min\' or \'max\' must not be None");
    if (min == nullptr) {
        return diopiClampMaxScalar(ctx, out, input, max);
    } else if (max == nullptr) {
        return diopiClampMinScalar(ctx, out, input, min);
    }
    auto check = clampScalarCheck(ctx, input, out, min, max);
    if (diopiSuccess != check) {
        return check;
    }
    DiopiTensor minTensorTmp;
    DiopiTensor maxTensorTmp;
    makeTensorFromScalar(ctx, min, minTensorTmp);
    makeTensorFromScalar(ctx, max, maxTensorTmp);
    diopiTensorHandle_t minTensor = minTensorTmp.tensorHandle();
    diopiTensorHandle_t maxTensor = maxTensorTmp.tensorHandle();
    return clampCommon(ctx, input, out, minTensor, maxTensor);
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    DIOPI_CHECK(min != nullptr || max != nullptr, "At least one of \'min\' or \'max\' must not be None");
    auto check = clampTensorCheck(ctx, input, out);
    if (diopiSuccess != check) {
        return check;
    }
    if (min == nullptr) {
        return diopiClampMax(ctx, out, input, max);
    } else if (max == nullptr) {
        return diopiClampMin(ctx, out, input, min);
    }
    return clampCommon(ctx, input, out, min, max);
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    DiopiTensor maxTensorTmp;
    makeTensorFromScalar(ctx, max, maxTensorTmp);
    diopiTensorHandle_t maxTensor = maxTensorTmp.tensorHandle();
    return clampCommon(ctx, input, input, nullptr, maxTensor);
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, input, nullptr, max);
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    auto check = clampScalarCheck(ctx, input, out, nullptr, max);
    if (diopiSuccess != check) {
        return check;
    }
    DiopiTensor maxTensorTmp;
    makeTensorFromScalar(ctx, max, maxTensorTmp);
    diopiTensorHandle_t maxTensor = maxTensorTmp.tensorHandle();
    return clampCommon(ctx, input, out, nullptr, maxTensor);
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    auto check = clampTensorCheck(ctx, input, out);
    if (diopiSuccess != check) {
        return check;
    }
    return clampCommon(ctx, input, out, nullptr, max);
}

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    DiopiTensor minTensorTmp;
    makeTensorFromScalar(ctx, min, minTensorTmp);
    diopiTensorHandle_t minTensor = minTensorTmp.tensorHandle();
    return clampCommon(ctx, input, input, minTensor, nullptr);
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    return clampCommon(ctx, input, input, min, nullptr);
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    auto check = clampScalarCheck(ctx, input, out, min, nullptr);
    if (diopiSuccess != check) {
        return check;
    }
    DiopiTensor minTensorTmp;
    makeTensorFromScalar(ctx, min, minTensorTmp);
    diopiTensorHandle_t minTensor = minTensorTmp.tensorHandle();
    return clampCommon(ctx, input, out, minTensor, nullptr);
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    auto check = clampTensorCheck(ctx, input, out);
    if (diopiSuccess != check) {
        return check;
    }
    return clampCommon(ctx, input, out, min, nullptr);
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
