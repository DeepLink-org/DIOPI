/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

/**
 * @brief Computes the inverse error function of input tensor.
 */

DIOPI_API diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor *> tensorsVecPtr{&inputTensor, &outTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, tensorsVecPtr, supportedDtypes));

    DiopiTensor inputCastedTensor = *tensorsVecPtr[0];
    DiopiTensor outCastedTensor = *tensorsVecPtr[1];
    CnnlTensorDesc inputDesc(inputCastedTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outCastedTensor, CNNL_LAYOUT_ARRAY);

    cnnlComputationPreference_t computePrefer = CNNL_COMPUTATION_HIGH_PRECISION;
    DIOPI_CALLCNNL(cnnlErfinv(handle, computePrefer, inputDesc.get(), inputCastedTensor.data(), outDesc.get(), outCastedTensor.data()));
    if (outCastedTensor.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outCastedTensor));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(diopiErfinv(ctx, input, input));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
