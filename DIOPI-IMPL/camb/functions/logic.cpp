#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t
LogicScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto out_tensor = makeTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    auto other_t_tensor = makeTensor(other_t);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);

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
                               out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t LogicInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t other_t;
    diopiSize_t input_shape;
    DIOPI_CALL(diopiGetTensorShape(input, &input_shape));
    DIOPI_CALL(diopiRequireTensor(ctx, &other_t, &input_shape, nullptr, input_tensor.dtype(), diopi_device));
    DIOPI_CALL(diopiFill(ctx, other_t, other));
    auto other_t_tensor = makeTensor(other_t);
    CnnlTensorDesc other_t_desc(other_t_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_t_desc.get(), input_desc.get(), &workspace_size));
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
                               input_desc.get(),
                               input_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t
Logic(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto other_tensor = makeTensor(other);
    auto out_tensor = makeTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

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
                               out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t LogicInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, cnnlLogicOp_t logic_op) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = makeTensor(input);
    auto other_tensor = makeTensor(other);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetLogicOpWorkspaceSize(handle, input_desc.get(), other_desc.get(), input_desc.get(), &workspace_size));
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
                               input_desc.get(),
                               input_tensor.data()));
    return diopiSuccess;
}

// ge
DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GE));
}

DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GE));
}

DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_GE));
}

DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_GE));
}

// gt
DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_GT));
}

DIOPI_API diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_GT));
}

DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_GT));
}

DIOPI_API diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_GT));
}

// le
DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LE));
}

DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LE));
}

DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_LE));
}

DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_LE));
}

// lt
DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_LT));
}

DIOPI_API diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_LT));
}

DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_LT));
}

DIOPI_API diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_LT));
}

// ne
DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_NE));
}

DIOPI_API diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_NE));
}

DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_NE));
}

DIOPI_API diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_NE));
}

// eq
DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicScalar(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
}

DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(LogicInpScalar(ctx, input, other, CNNL_LOGIC_OP_EQ));
}

DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(Logic(ctx, out, input, other, CNNL_LOGIC_OP_EQ));
}

DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(LogicInp(ctx, input, other, CNNL_LOGIC_OP_EQ));
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
