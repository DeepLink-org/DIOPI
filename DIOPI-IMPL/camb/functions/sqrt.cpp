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

static diopiError_t sqrt(diopiContextHandle_t ctx, DiopiTensor& output, DiopiTensor input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<DiopiTensor*> pTensors{&input};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor output_tmp = output;
    if (input.dtype() != output.dtype()) {
        output_tmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc desc(input, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlSqrt_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, desc.get(), input.data(), desc.get(), output_tmp.data()));
    if (output_tmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, output_tmp));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DIOPI_CALL(sqrt(ctx, input_tensor, input_tensor));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(sqrt(ctx, output_tensor, input_tensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
