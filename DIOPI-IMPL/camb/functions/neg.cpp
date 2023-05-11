
#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlNegTensor(handle, input_desc.get(), input_tensor_tmp.data(), out_desc.get(), out_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(diopiNeg(ctx, input, input));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
