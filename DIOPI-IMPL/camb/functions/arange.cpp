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

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto out_tensor = DiopiTensor(out);
    DiopiTensor out32_tensor = out_tensor;
    if (diopi_dtype_int64 == out_tensor.dtype()) {
        dataTypeCast(ctx, out32_tensor, diopi_dtype_int32);
    }
    CnnlTensorDesc outDesc(out32_tensor, CNNL_LAYOUT_ARRAY);

    cnnlDataType_t dtype;
    CnnlDataType::convertToCnnlType(&dtype, out32_tensor.dtype());
    if (CnnlDataType::isInteger(dtype)) {
        DIOPI_CALLCNNL(cnnlArange_v2(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION, &(start->ival), &(step->ival), outDesc.get(), out32_tensor.data()));
        if (out32_tensor.dtype() != out_tensor.dtype()) {
            dataTypeCast(ctx, out_tensor, out32_tensor);
        }
    } else if (CnnlDataType::isFloat(dtype)) {
        float start_val = start->fval;
        float step_val = step->fval;
        DIOPI_CALLCNNL(cnnlArange_v2(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION, &(start_val), &(step_val), outDesc.get(), out32_tensor.data()));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
