/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    int64_t numel = 0;
    diopiGetTensorNumel(src, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    AscendTensor inputAt(src), outAt(dest);
    if (src != dest) {
        if (inputAt.dtype() == outAt.dtype() && 1 != numel) {
            AclOpRunner<8, 1>("ViewCopy", ctx)
                .addInputWithoutContiguous(dest)
                .addConstInput(outAt.shape())
                .addConstInput(outAt.stride())
                .addConstInput(0, diopi_dtype_int64)
                .addInputWithoutContiguous(src)
                .addConstInput(inputAt.shape())
                .addConstInput(inputAt.stride())
                .addConstInput(0, diopi_dtype_int64)
                .addOutputWithoutContiguous(dest)
                .run();
        } else {
            diopiCastDtype(ctx, dest, src);
        }
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
