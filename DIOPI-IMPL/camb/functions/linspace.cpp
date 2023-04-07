#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    auto handle = cnnlHandlePool.get(ctx);
    DiopiTensor out_tensor(out);

    float start_value, end_value;

    cnnlDataType_t start_type, end_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&start_type, start->stype));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&end_type, end->stype));

    if (CnnlDataType::isFloat(start_type)) {
        start_value = start->fval;
    } else if (CnnlDataType::isInteger(start_type)) {
        start_value = start->ival;
    } else {
        return diopiDtypeNotSupported;
    }

    if (CnnlDataType::isFloat(end_type)) {
        end_value = end->fval;
    } else if (CnnlDataType::isInteger(start_type)) {
        end_value = end->ival;
    } else {
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlLinspace(handle, start_value, end_value, out_desc.get(), out_tensor.data()));
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
