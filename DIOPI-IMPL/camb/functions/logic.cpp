/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <string.h>

#include <set>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t Logic(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                             cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &other_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor out_tensor_temp = out_tensor;
    if (out_tensor.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op,
                               input_desc.get(),
                               input_tensor.data(),
                               other_desc.get(),
                               other_tensor.data(),
                               workspace,
                               workspace_size,
                               out_desc.get(),
                               out_tensor_temp.data()));
    if (out_tensor_temp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    return diopiSuccess;
}

diopiError_t LogicInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op) {
    DIOPI_CALL(Logic(ctx, input, input, other, logic_op));
}

diopiError_t LogicScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                   cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    DiopiTensor other_t_tensor(other_t);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &other_t_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor out_tensor_temp = out_tensor;
    if (out_tensor.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_t_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlLogicOp(handle,
                               logic_op,
                               input_desc.get(),
                               input_tensor.data(),
                               other_t_desc.get(),
                               other_t_tensor.data(),
                               workspace,
                               workspace_size,
                               out_desc.get(),
                               out_tensor_temp.data()));
    if (out_tensor_temp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    return diopiSuccess;
}

diopiError_t LogicInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op) {
    DIOPI_CALL(LogicScalar(ctx, input, input, other, logic_op));
}

// ge
diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GE));
}

diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GE));
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_GE));
}

diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_GE));
}

// gt
diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GT));
}

diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GT));
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_GT));
}

diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_GT));
}

// le
diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LE));
}

diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LE));
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_LE));
}

diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_LE));
}

// lt
diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LT));
}

diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LT));
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_LT));
}

diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_LT));
}

// ne
diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_NE));
}

diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_NE));
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_NE));
}

diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_NE));
}

// eq
diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
}

diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_EQ));
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
}

diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_EQ));
}

//  logical_and
diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_AND));
}

diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_AND));
}

// logical_or
diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_OR));
}

diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_OR));
}

// logical_not
diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Logic(ctx, out, input, input, CNNL_LOGIC_OP_NOT));
}

diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { DIOPI_CALL(LogicInp(ctx, input, input, CNNL_LOGIC_OP_NOT)); }

}  // extern "C"

}  // namespace camb
}  // namespace impl
