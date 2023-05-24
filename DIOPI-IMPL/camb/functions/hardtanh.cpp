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

    std::vector<DiopiTensor*> pTensors{&inputTensor, &outTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    double min = DiopiDataType::isInteger(minVal->stype) ? minVal->ival : minVal->fval;
    double max = DiopiDataType::isInteger(maxVal->stype) ? maxVal->ival : maxVal->fval;

    DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), inputTensor.data(), float(max), float(min), outDesc.get(), outTensor.data()));
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
    DiopiTensor gradInTensor(gradInput);
    DiopiTensor gradOutTensor(gradOutput);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &gradOutTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradoutDesc(gradOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradinDesc(gradInTensor, CNNL_LAYOUT_ARRAY);

    double min = DiopiDataType::isInteger(minVal->stype) ? minVal->ival : minVal->fval;
    double max = DiopiDataType::isInteger(maxVal->stype) ? maxVal->ival : maxVal->fval;

    DIOPI_CALLCNNL(cnnlHardtanhBackward(
        handle, inputDesc.get(), inputTensor.data(), gradoutDesc.get(), gradOutTensor.data(), max, min, gradinDesc.get(), gradInTensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
