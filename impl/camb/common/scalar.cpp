/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, DiopiTensor& out) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    if (scalar->stype == diopi_dtype_int64) {
        int32_t val = static_cast<int32_t>(scalar->ival);
        out = requiresTensor(ctx, sSize, diopi_dtype_int32);
        CnnlTensorDesc descOut(out, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &val, descOut.get(), out.data()));
        return diopiSuccess;
    } else if (scalar->stype == diopi_dtype_float64) {
        float val = static_cast<float>(scalar->fval);
        out = requiresTensor(ctx, sSize, diopi_dtype_float32);
        CnnlTensorDesc descOut(out, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &val, descOut.get(), out.data()));
        return diopiSuccess;
    } else {
        setLastErrorString("%s", "salar dtype is not float64 or int64");
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
