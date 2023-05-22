/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>
#include <set>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t logic(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                             cnnlLogicOp_t logicOp) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &otherTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outTensorTemp = outTensor;
    if (outTensor.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logicOp,
                               inputDesc.get(),
                               inputTensor.data(),
                               otherDesc.get(),
                               otherTensor.data(),
                               workspace,
                               workspaceSize,
                               outDesc.get(),
                               outTensorTemp.data()));
    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

diopiError_t logicInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logicOp) {
    DIOPI_CALL(logic(ctx, input, input, other, logicOp));
    return diopiSuccess;
}

diopiError_t logicScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                   cnnlLogicOp_t logicOp) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    diopiTensorHandle_t otherT;
    diopiSize_t inputShape;
    DIOPI_CALL(diopiGetTensorShape(input, &inputShape));
    DIOPI_CALL(diopiRequireTensor(ctx, &otherT, &inputShape, nullptr, inputTensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, otherT, other));
    DiopiTensor otherTTensor(otherT);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &otherTTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outTensorTemp = outTensor;
    if (outTensor.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherTDesc(otherTTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, inputDesc.get(), otherTDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logicOp,
                               inputDesc.get(),
                               inputTensor.data(),
                               otherTDesc.get(),
                               otherTTensor.data(),
                               workspace,
                               workspaceSize,
                               outDesc.get(),
                               outTensorTemp.data()));
    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

diopiError_t logicInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logicOp) {
    DIOPI_CALL(logicScalar(ctx, input, input, other, logicOp));
    return diopiSuccess;
}

// ge
diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GE));
    return diopiSuccess;
}

diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GE));
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_GE));
    return diopiSuccess;
}

diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_GE));
    return diopiSuccess;
}

// gt
diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GT));
    return diopiSuccess;
}

diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GT));
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_GT));
    return diopiSuccess;
}

diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_GT));
    return diopiSuccess;
}

// le
diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LE));
    return diopiSuccess;
}

diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LE));
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_LE));
    return diopiSuccess;
}

diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_LE));
    return diopiSuccess;
}

// lt
diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LT));
    return diopiSuccess;
}

diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LT));
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_LT));
    return diopiSuccess;
}

diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_LT));
    return diopiSuccess;
}

// ne
diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicScalar(ctx, out, input, other, CNNL_LOGIC_OP_NE));
    return diopiSuccess;
}

diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicInpScalar(ctx, input, other, CNNL_LOGIC_OP_NE));
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_NE));
    return diopiSuccess;
}

diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_NE));
    return diopiSuccess;
}

// eq
diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicScalar(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
    return diopiSuccess;
}

diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(logicInpScalar(ctx, input, other, CNNL_LOGIC_OP_EQ));
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
    return diopiSuccess;
}

diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_EQ));
    return diopiSuccess;
}

//  logical_and
diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_AND));
    return diopiSuccess;
}

diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_AND));
    return diopiSuccess;
}

// logical_or
diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logic(ctx, out, input, other, CNNL_LOGIC_OP_OR));
    return diopiSuccess;
}

diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(logicInp(ctx, input, other, CNNL_LOGIC_OP_OR));
    return diopiSuccess;
}

// logical_not
diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(logic(ctx, out, input, input, CNNL_LOGIC_OP_NOT));
    return diopiSuccess;
}

diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(logicInp(ctx, input, input, CNNL_LOGIC_OP_NOT));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
