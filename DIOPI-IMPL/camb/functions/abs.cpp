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

static diopiError_t abs(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor& output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<DiopiTensor*> pTensors{&input};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor output_tmp = output;
    if (input.dtype() != output.dtype()) {
        output_tmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc input_desc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_tmp_desc(output_tmp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlAbs(handle, input_desc.get(), input.data(), output_tmp_desc.get(), output_tmp.data()));
    if (output_tmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, output_tmp));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DIOPI_CALL(abs(ctx, input_tensor, input_tensor));
    return diopiSuccess;
}

extern "C" diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(abs(ctx, input_tensor, output_tensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
