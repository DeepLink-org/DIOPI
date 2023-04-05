#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t Log(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, cnnlLogBase_t log_base) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlLog(handle, log_base, input_desc.get(), input_tensor_tmp.data(), out_desc.get(), out_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    return diopiSuccess;
}

DIOPI_API diopiError_t LogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, cnnlLogBase_t log_base) {
    DIOPI_CALL(Log(ctx, input, input, log_base));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(LogInp(ctx, input, CNNL_LOG_E));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Log(ctx, out, input, CNNL_LOG_E));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(LogInp(ctx, input, CNNL_LOG_2));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Log(ctx, out, input, CNNL_LOG_2));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(LogInp(ctx, input, CNNL_LOG_10));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DIOPI_CALL(Log(ctx, out, input, CNNL_LOG_10));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
