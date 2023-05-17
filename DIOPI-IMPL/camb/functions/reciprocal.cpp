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

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    diopiDtype_t origin_dtype = input_tensor.dtype();
    std::vector<DiopiTensor*> pTensors{&input_tensor, &out_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlReciprocal(handle, input_desc.get(), input_tensor.data(), out_desc.get(), out_tensor.data()));

    if (origin_dtype == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, origin_dtype));
    }
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiReciprocal(ctx, input, input);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
