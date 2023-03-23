/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {
extern "C" {

diopiError_t bitwiseCommon(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, cnnlBitComputeOp_t optype) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto out_tensor = DiopiTensor(out);
    auto out32_tensor = out_tensor;
    if (diopi_dtype_int64 == out_tensor.dtype()) {
        out32_tensor = dataTypeCast(ctx, out_tensor, diopi_dtype_int32);
    }
    CnnlTensorDesc outDesc(out32_tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t input1 = const_cast<diopiTensorHandle_t>(input);
    auto input1_tensor = DiopiTensor(input1);
    if (input1_tensor.dtype() != out32_tensor.dtype()) {
        input1_tensor = dataTypeCast(ctx, input1_tensor, out32_tensor.dtype());
    }
    CnnlTensorDesc input1Desc(input1_tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t input2 = const_cast<diopiTensorHandle_t>(other);
    const void* input2_ptr = nullptr;
    CnnlTensorDesc input2Desc;
    cnnlTensorDescriptor_t input2_desc = nullptr;
    if (nullptr != other) {
        auto input2_tensor = DiopiTensor(input2);
        if (input2_tensor.dtype() != out32_tensor.dtype()) {
            input2_tensor = dataTypeCast(ctx, input2_tensor, out32_tensor.dtype());
        }
        input2_ptr = input2_tensor.data();
        input2Desc.set(input2_tensor, CNNL_LAYOUT_ARRAY);
        input2_desc = input2Desc.get();
    }

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetBitComputeWorkspaceSize(handle, input1Desc.get(), input2_desc, outDesc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlBitCompute_v2(
        handle, optype, input1Desc.get(), input1_tensor.data(), input2_desc, input2_ptr, outDesc.get(), out32_tensor.data(), workspace, workspace_size));
    if (out_tensor.dtype() != out32_tensor.dtype()) {
        dataTypeCast(ctx, out_tensor, out32_tensor);
    }

    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return bitwiseCommon(ctx, out, input, other, CNNL_CYCLE_BAND_OP);
}

diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return bitwiseCommon(ctx, input, input, other, CNNL_CYCLE_BAND_OP);
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t input2 = diopiTensorHandle_t(makeTensorFromScalar(ctx, other));
    return bitwiseCommon(ctx, out, input, diopiTensorHandle_t(input2), CNNL_CYCLE_BAND_OP);
}

diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t input2 = diopiTensorHandle_t(makeTensorFromScalar(ctx, other));
    return bitwiseCommon(ctx, input, input, input2, CNNL_CYCLE_BAND_OP);
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return bitwiseCommon(ctx, out, input, other, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return bitwiseCommon(ctx, input, input, other, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t input2 = diopiTensorHandle_t(makeTensorFromScalar(ctx, other));
    return bitwiseCommon(ctx, out, input, input2, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiTensorHandle_t input2 = diopiTensorHandle_t(makeTensorFromScalar(ctx, other));
    return bitwiseCommon(ctx, input, input, input2, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    return bitwiseCommon(ctx, out, input, nullptr, CNNL_BNOT_OP);
}

diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return bitwiseCommon(ctx, input, input, nullptr, CNNL_BNOT_OP); }

}  // extern "C"

}  // namespace camb
}  // namespace impl
