/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

static diopiError_t exp(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor& output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<DiopiTensor*> pTensors{&input};
    DIOPI_CHECK(input.shape() == output.shape(), "input shape should be same as output");
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor output_tmp = output;
    if (input.dtype() != output.dtype()) {
        output_tmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc desc(input, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExp_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, desc.get(), input.data(), desc.get(), output_tmp.data()));
    if (output.dtype() != output_tmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, output_tmp));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DIOPI_CALL(exp(ctx, input_tensor, input_tensor));
    return diopiSuccess;
}

extern "C" diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(exp(ctx, input_tensor, output_tensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
