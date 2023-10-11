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
    if (src != dest) {
        diopiDtype_t dstType, srcType;
        diopiGetTensorDtype(dest, &dstType);
        diopiGetTensorDtype(src, &srcType);
        if (dstType == srcType) {
            diopiSize_t dstSize, srcSize, dstStride, srcStride;
            diopiGetTensorShape(dest, &dstSize);
            diopiGetTensorShape(src, &srcSize);
            diopiGetTensorStride(dest, &dstStride);
            diopiGetTensorStride(src, &srcStride);
            AclOpRunner<8, 1>("ViewCopy", ctx)
                .addInputWithoutContiguous(dest)
                .addConstInput(dstSize)
                .addConstInput(dstStride)
                .addConstInput(0, diopi_dtype_int64)
                .addInputWithoutContiguous(src)
                .addConstInput(srcSize)
                .addConstInput(srcStride)
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
