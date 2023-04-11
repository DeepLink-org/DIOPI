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

static diopiError_t cos(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor& output) {
    auto handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    std::vector<DiopiTensor*> pTensors{&input};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor output_tmp = output;
    if (input.dtype() != output.dtype()) {
        output_tmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc input_desc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_tmp_desc(output_tmp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, input_desc.get(), input.data(), output_tmp_desc.get(), output_tmp.data()));
    if (output_tmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, output_tmp));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DIOPI_CALL(cos(ctx, input_tensor, input_tensor));
    return diopiSuccess;
}

extern "C" diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(cos(ctx, input_tensor, output_tensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
