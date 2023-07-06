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
    auto inputTensor = DiopiTensor(input);
    std::vector<DiopiTensor*> pTensors{&input};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor outputTmp = output;
    if (input.dtype() != output.dtype()) {
        outputTmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputTmpDesc(outputTmp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, inputDesc.get(), input.data(), outputTmpDesc.get(), outputTmp.data()));
    if (outputTmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, outputTmp));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DIOPI_CALL(cos(ctx, inputTensor, inputTensor));
    return diopiSuccess;
}

extern "C" diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DIOPI_CALL(cos(ctx, inputTensor, outputTensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
