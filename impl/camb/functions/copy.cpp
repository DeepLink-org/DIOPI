/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    if (src == dest) {
        // the same address of pointers, return earlier
        return diopiSuccess;
    }

    // TODO(waiting for dispatch): support broadcast, dealing with uncontiguous
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor destTr(dest);
    DiopiTensor srcTr(src);

    if (srcTr.dtype() != destTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, srcTr, destTr.dtype()));
    }

    CnnlTensorDesc inputDesc(destTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(srcTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCopy(handle, srcDesc.get(), srcTr.data(), inputDesc.get(), destTr.data()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
