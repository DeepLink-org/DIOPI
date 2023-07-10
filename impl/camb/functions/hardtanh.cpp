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

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* minVal,
                           const diopiScalar_t* maxVal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor outTensorTmp = outTensor;

    std::vector<DiopiTensor*> pTensors{&inputTensor, &outTensorTmp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    auto min = DiopiDataType::isInteger(minVal->stype) ? minVal->ival : minVal->fval;
    auto max = DiopiDataType::isInteger(maxVal->stype) ? maxVal->ival : maxVal->fval;
    min = min > max ? max : min;
    // DIOPI_CHECK(max > min, "assert max.val > min.val");

    DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), inputTensor.data(), max, min, outDesc.get(), outTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    DIOPI_CALL(diopiHardtanh(ctx, input, input, minVal, maxVal));
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                   const diopiScalar_t* minVal, const diopiScalar_t* maxVal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutTensor(gradOutput);
    DiopiTensor gradInputTensorTmp(gradInput);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &gradInputTensorTmp, &gradOutTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutDesc(gradOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInDesc(gradInputTensorTmp, CNNL_LAYOUT_ARRAY);

    auto min = DiopiDataType::isInteger(minVal->stype) ? minVal->ival : minVal->fval;
    auto max = DiopiDataType::isInteger(maxVal->stype) ? maxVal->ival : maxVal->fval;
    min = min > max ? max : min;
    // DIOPI_CHECK(max > min, "assert max.val > min.val");

    DIOPI_CALLCNNL(cnnlHardtanhBackward(
        handle, inputDesc.get(), inputTensor.data(), gradOutDesc.get(), gradOutTensor.data(), max, min, gradInDesc.get(), gradInputTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
