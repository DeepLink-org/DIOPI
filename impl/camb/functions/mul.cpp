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

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);

    DiopiTensor outTensorTmp = outTensor;
    if ((outTensor.dtype() != diopi_dtype_float16) && (outTensor.dtype() != diopi_dtype_float32)) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, diopi_dtype_float32));
    }
    DIOPI_CALL(dataTypeCast(ctx, inputTensor, outTensorTmp.dtype()));
    DIOPI_CALL(dataTypeCast(ctx, otherTensor, outTensorTmp.dtype()));

    DiopiTensor bcastInputTensor;
    broadcastHelper(ctx, inputTensor, outTensorTmp, &bcastInputTensor);
    DiopiTensor bcastOtherTensor;
    broadcastHelper(ctx, otherTensor, outTensorTmp, &bcastOtherTensor);

    CnnlTensorDesc bcastInputDesc(bcastInputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc bcastOtherDesc(bcastOtherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    cnnlTensorDescriptor_t inputDescs[] = {bcastInputDesc.get(), bcastOtherDesc.get()};
    const void* inputs[] = {bcastInputTensor.data(), bcastOtherTensor.data()};

    DIOPI_CALLCNNL(cnnlMulN(handle, inputDescs, inputs, 2, outDesc.get(), outTensorTmp.data()))
    if (outTensorTmp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(diopiMul(ctx, input, input, other));
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor otherTensorTmp;
    makeTensorFromScalar(ctx, other, otherTensorTmp);
    auto otherTensor = otherTensorTmp.tensorHandle();
    DIOPI_CALL(diopiMul(ctx, out, input, diopiTensorHandle_t(otherTensor)));
    return diopiSuccess;
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(diopiMulScalar(ctx, input, input, other));
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
