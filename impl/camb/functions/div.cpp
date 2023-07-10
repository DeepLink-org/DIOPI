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

diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      diopiRoundMode_t roundingMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);

    DiopiTensor outTensorTemp = outTensor;
    if ((outTensor.dtype() != diopi_dtype_float16) && (outTensor.dtype() != diopi_dtype_float32)) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, diopi_dtype_float32));
    } else {
        outTensorTemp = DiopiTensor(out);
    }

    DIOPI_CALL(dataTypeCast(ctx, inputTensor, outTensorTemp.dtype()));
    DIOPI_CALL(dataTypeCast(ctx, otherTensor, outTensorTemp.dtype()));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetDivWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    workspace = requiresBuffer(ctx, workspaceSize).data();

    cnnlDiv_v2(handle,
               CNNL_COMPUTATION_HIGH_PRECISION,
               inputDesc.get(),
               inputTensor.data(),
               otherDesc.get(),
               otherTensor.data(),
               workspace,
               workspaceSize,
               outDesc.get(),
               outTensorTemp.data());
    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    DIOPI_CALL(diopiDiv(ctx, input, input, other, roundingMode));
    return diopiSuccess;
}

diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            diopiRoundMode_t roundingMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensorTmp;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensorTmp));
    auto otherTensor = otherTensorTmp.tensorHandle();
    DiopiTensor outTensor(out);
    DIOPI_CALL(diopiDiv(ctx, out, input, diopiTensorHandle_t(otherTensor), roundingMode));
    return diopiSuccess;
}
diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t roundingMode) {
    DIOPI_CALL(diopiDivScalar(ctx, input, input, other, roundingMode));
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
