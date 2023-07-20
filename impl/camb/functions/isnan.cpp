/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor inputTensorTmp = inputTensor;

    std::vector<DiopiTensor *> tensor{&inputTensorTmp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensor, supportedDtypes));

    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlIsNan(handle, inputDesc.get(), inputTensorTmp.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
