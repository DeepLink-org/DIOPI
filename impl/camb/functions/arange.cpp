/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor out32Tensor = outTensor;
    if (diopi_dtype_int64 == outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out32Tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc outDesc(out32Tensor, CNNL_LAYOUT_ARRAY);

    cnnlDataType_t dtype;
    CnnlDataType::convertToCnnlType(&dtype, out32Tensor.dtype());
    if (CnnlDataType::isInteger(dtype)) {
        DIOPI_CALLCNNL(cnnlArange_v2(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION, &(start->ival), &(step->ival), outDesc.get(), out32Tensor.data()));
        if (out32Tensor.dtype() != outTensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, outTensor, out32Tensor));
        }
    } else if (CnnlDataType::isFloatPoint(dtype)) {
        float startVal = DiopiDataType::isInteger(start->stype) ? start->ival : start->fval;
        float stepVal = DiopiDataType::isInteger(step->stype) ? step->ival : step->fval;
        DIOPI_CALLCNNL(cnnlArange_v2(handle, CNNL_COMPUTATION_ULTRAHIGH_PRECISION, &(startVal), &(stepVal), outDesc.get(), out32Tensor.data()));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
