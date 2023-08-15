#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    auto handle = cnnlHandlePool.get(ctx);
    DiopiTensor outTensor(out);

    float startValue, endValue;

    cnnlDataType_t startType, endType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&startType, start->stype));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&endType, end->stype));

    if (CnnlDataType::isFloatPoint(startType)) {
        startValue = start->fval;
    } else if (CnnlDataType::isInteger(startType)) {
        startValue = start->ival;
    } else {
        return diopiDtypeNotSupported;
    }

    if (CnnlDataType::isFloatPoint(endType)) {
        endValue = end->fval;
    } else if (CnnlDataType::isInteger(startType)) {
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
