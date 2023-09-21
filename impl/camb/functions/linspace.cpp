#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    auto handle = cnnlHandlePool.get(ctx);
    DiopiTensor outTensor(out);

    float startValue, endValue;

    diopiDtype_t startType = start->stype;
    diopiDtype_t endType = end->stype;
    if (DiopiDataType::isFloatPoint(startType)) {
        startValue = start->fval;
    } else if (DiopiDataType::isInteger(startType)) {
        startValue = start->ival;
    } else {
        return diopiDtypeNotSupported;
    }

    if (DiopiDataType::isFloatPoint(endType)) {
        endValue = end->fval;
    } else if (DiopiDataType::isInteger(startType)) {
        endValue = end->ival;
    } else {
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlLinspace(handle, startValue, endValue, outDesc.get(), outTensor.data()));
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
