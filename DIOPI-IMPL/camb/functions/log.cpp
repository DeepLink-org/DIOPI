#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t log(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, cnnlLogBase_t logBase) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlLog(handle, logBase, inputDesc.get(), inputTensorTmp.data(), outDesc.get(), outTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    return diopiSuccess;
}

diopiError_t logInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, cnnlLogBase_t logBase) {
    DIOPI_CALL(log(ctx, input, input, logBase));
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(logInp(ctx, input, CNNL_LOG_E));
    return diopiSuccess;
}

diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(log(ctx, out, input, CNNL_LOG_E));
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(logInp(ctx, input, CNNL_LOG_2));
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(log(ctx, out, input, CNNL_LOG_2));
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(logInp(ctx, input, CNNL_LOG_10));
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(log(ctx, out, input, CNNL_LOG_10));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
