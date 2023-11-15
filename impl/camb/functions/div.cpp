/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t roundingMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);
    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
    cnnlComputationPreference_t preferFloor = CNNL_COMPUTATION_ULTRAHIGH_PRECISION;

    DiopiTensor outTensorTemp = DiopiTensor(out);
    std::vector<DiopiTensor *> pTensors{&inputTensor, &otherTensor, &outTensorTemp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);
    size_t workspaceSize = 0;
    void *workspace = nullptr;

    switch (roundingMode) {
        case RoundModeFloor:
            DIOPI_CALL_CNNL(cnnlGetFloorDivWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
            workspace = requiresBuffer(ctx, workspaceSize).data();
            DIOPI_CALL_CNNL(cnnlFloorDiv_v2(handle,
                                            preferFloor,
                                            inputDesc.get(),
                                            inputTensor.data(),
                                            otherDesc.get(),
                                            otherTensor.data(),
                                            outDesc.get(),
                                            outTensorTemp.data(),
                                            workspace,
                                            workspaceSize));
            break;
        case RoundModeTrunc:
            DIOPI_CALL_CNNL(cnnlGetFloorDivTruncWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
            workspace = requiresBuffer(ctx, workspaceSize).data();
            DIOPI_CALL_CNNL(cnnlFloorDivTrunc(handle,
                                              prefer,
                                              inputDesc.get(),
                                              inputTensor.data(),
                                              otherDesc.get(),
                                              otherTensor.data(),
                                              outDesc.get(),
                                              outTensorTemp.data(),
                                              workspace,
                                              workspaceSize));
            break;
        case RoundModeNone:
            DIOPI_CALL_CNNL(cnnlGetDivWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
            workspace = requiresBuffer(ctx, workspaceSize).data();
            DIOPI_CALL_CNNL(cnnlDiv_v2(handle,
                                       prefer,
                                       inputDesc.get(),
                                       inputTensor.data(),
                                       otherDesc.get(),
                                       otherTensor.data(),
                                       workspace,
                                       workspaceSize,
                                       outDesc.get(),
                                       outTensorTemp.data()));

            break;
        default:
            break;
    }
    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    DIOPI_CALL(diopiDiv(ctx, input, input, other, roundingMode));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                      diopiRoundMode_t roundingMode) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DiopiTensor outTensor(out);
    DIOPI_CALL(diopiDiv(ctx, out, input, diopiTensorHandle_t(otherTensor), roundingMode));
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other, diopiRoundMode_t roundingMode) {
    DIOPI_CALL(diopiDivScalar(ctx, input, input, other, roundingMode));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
