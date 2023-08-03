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

diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    // std::vector<DiopiTensor*> pTensors{&inputTensor};
    // std::set<diopiDtype_t> supportedDtypes{diopi_dtype_complex64};
    // DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // DiopiTensor outTensorTemp = outTensor;
    if (inputTensor.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, inputTensor.dtype()));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlSign(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));

    if (outTensor.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensor));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DIOPI_CALL(diopiSgn(ctx, input, input));
    return diopiSuccess;
}
}
}  // namespace camb
}  // namespace impl
