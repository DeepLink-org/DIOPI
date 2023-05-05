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

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    if (src == input) {
        // the same address of pointers, return earlier
        return diopiSuccess;
    }

    // TODO(waiting for dispatch): support broadcast, dealing with uncontiguous
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor dest_tr(input);
    DiopiTensor src_tr(src);



    if (src_tr.dtype() != dest_tr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, dest_tr, src_tr));
    }

    CnnlTensorDesc input_desc(dest_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc src_desc(src_tr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCopy(handle, src_desc.get(), src_tr.data(), input_desc.get(), dest_tr.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
