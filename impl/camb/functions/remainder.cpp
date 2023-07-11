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

DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);

    DiopiTensor outTensorTemp = outTensor;
    std::vector<DiopiTensor *> pTensors{&inputTensor, &otherTensor, &outTensorTemp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetFloorModWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlFloorMod(
        handle, inputDesc.get(), inputTensor.data(), otherDesc.get(), otherTensor.data(), outDesc.get(), outTensorTemp.data(), workspace, workspaceSize));

    if (outTensor.dtype() != outTensorTemp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    diopiTensorHandle_t otherTensorPtr = otherTensor.tensorHandle();
    DIOPI_CALL(diopiRemainderTensor(ctx, out, input, otherTensorPtr));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *input, diopiConstTensorHandle_t other) {
    DiopiTensor inputTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, input, inputTensor));
    diopiTensorHandle_t inputTensorPtr = inputTensor.tensorHandle();
    DIOPI_CALL(diopiRemainderTensor(ctx, out, inputTensorPtr, other));
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl