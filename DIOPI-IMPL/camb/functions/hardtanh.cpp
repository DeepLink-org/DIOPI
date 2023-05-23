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

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val,
                           const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &out_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    double min = DiopiDataType::isInteger(min_val->stype) ? min_val->ival : min_val->fval;
    double max = DiopiDataType::isInteger(max_val->stype) ? max_val->ival : max_val->fval;

    DIOPI_CALLCNNL(cnnlHardtanh(handle, inputDesc.get(), input_tensor.data(), float(max), float(min), outDesc.get(), out_tensor.data()));
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    DIOPI_CALL(diopiHardtanh(ctx, input, input, min_val, max_val));
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                   diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor grad_in_tensor(grad_input);
    DiopiTensor grad_out_tensor(grad_output);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &grad_out_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradoutDesc(grad_out_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradinDesc(grad_in_tensor, CNNL_LAYOUT_ARRAY);

    double min = DiopiDataType::isInteger(min_val->stype) ? min_val->ival : min_val->fval;
    double max = DiopiDataType::isInteger(max_val->stype) ? max_val->ival : max_val->fval;

    DIOPI_CALLCNNL(cnnlHardtanhBackward(
        handle, inputDesc.get(), input_tensor.data(), gradoutDesc.get(), grad_out_tensor.data(), max, min, gradinDesc.get(), grad_in_tensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
