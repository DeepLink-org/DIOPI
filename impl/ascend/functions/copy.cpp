/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
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
                .addInput(dest)
                .addConstInput(dstSize, diopi_dtype_int32)
                .addConstInput(dstStride, diopi_dtype_int32)
                .addConstInput<int>(0)
                .addInput(src)
                .addConstInput(srcSize, diopi_dtype_int32)
                .addConstInput(srcStride, diopi_dtype_int32)
                .addConstInput<int>(0)
                .addOutput(dest)
                .run();
        } else {
            diopiCastDtype(ctx, dest, src);
        }
    }
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
