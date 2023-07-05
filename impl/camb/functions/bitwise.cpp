/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
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

    DiopiTensor outTensor(out);
    auto out32Tensor = outTensor;
    if (diopi_dtype_int64 == outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out32Tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc outDesc(out32Tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t input1 = const_cast<diopiTensorHandle_t>(input);
    DiopiTensor input1Tensor(input1);
    if (input1Tensor.dtype() != out32Tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, input1Tensor, out32Tensor.dtype()));
    }
    CnnlTensorDesc input1Desc(input1Tensor, CNNL_LAYOUT_ARRAY);

    diopiTensorHandle_t input2 = const_cast<diopiTensorHandle_t>(other);
    const void* input2Ptr = nullptr;
    CnnlTensorDesc input2Desc;
    cnnlTensorDescriptor_t input2DescTmp = nullptr;
    if (nullptr != other) {
        DiopiTensor input2Tensor(input2);
        if (input2Tensor.dtype() != out32Tensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, input2Tensor, out32Tensor.dtype()));
        }
        input2Ptr = input2Tensor.data();
        input2Desc.set(input2Tensor, CNNL_LAYOUT_ARRAY);
        input2DescTmp = input2Desc.get();
    }

    size_t workspaceSize(0);
    DIOPI_CALLCNNL(cnnlGetBitComputeWorkspaceSize(handle, input1Desc.get(), input2DescTmp, outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlBitCompute_v2(
        handle, optype, input1Desc.get(), input1Tensor.data(), input2DescTmp, input2Ptr, outDesc.get(), out32Tensor.data(), workspace, workspaceSize));
    if (outTensor.dtype() != out32Tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, out32Tensor));
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
    DiopiTensor otherTensor;
    makeTensorFromScalar(ctx, other, otherTensor);
    diopiTensorHandle_t input2 = otherTensor.tensorHandle();
    return bitwiseCommon(ctx, out, input, diopiTensorHandle_t(input2), CNNL_CYCLE_BAND_OP);
}

diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor otherTensor;
    makeTensorFromScalar(ctx, other, otherTensor);
    diopiTensorHandle_t input2 = otherTensor.tensorHandle();
    return bitwiseCommon(ctx, input, input, input2, CNNL_CYCLE_BAND_OP);
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    return bitwiseCommon(ctx, out, input, other, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    return bitwiseCommon(ctx, input, input, other, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor otherTensor;
    makeTensorFromScalar(ctx, other, otherTensor);
    diopiTensorHandle_t input2 = otherTensor.tensorHandle();
    return bitwiseCommon(ctx, out, input, input2, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor otherTensor;
    makeTensorFromScalar(ctx, other, otherTensor);
    diopiTensorHandle_t input2 = otherTensor.tensorHandle();
    return bitwiseCommon(ctx, input, input, input2, CNNL_CYCLE_BOR_OP);
}

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    return bitwiseCommon(ctx, out, input, nullptr, CNNL_BNOT_OP);
}

diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return bitwiseCommon(ctx, input, input, nullptr, CNNL_BNOT_OP); }

}  // extern "C"

}  // namespace camb
}  // namespace impl
