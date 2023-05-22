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

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    if (src == dest) {
        // the same address of pointers, return earlier
        return diopiSuccess;
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor dest_tr(dest);
    DiopiTensor src_tr(src);

    if (src_tr.dtype() != dest_tr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, src_tr, dest_tr.dtype()));
    }

    DiopiTensor bcast_src_tr;
    DiopiTensor bcast_dest_tr;
    DiopiTensor target_tr = src_tr.numel() > dest_tr.numel() ? src_tr : dest_tr;

    DIOPI_CALL(broadcastHelper(ctx, src_tr, target_tr, &bcast_src_tr));
    DIOPI_CALL(broadcastHelper(ctx, dest_tr, target_tr, &bcast_dest_tr));

    CnnlTensorDesc input_desc(bcast_dest_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc src_desc(bcast_src_tr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCopy(handle, src_desc.get(), bcast_src_tr.data(), input_desc.get(), bcast_dest_tr.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
