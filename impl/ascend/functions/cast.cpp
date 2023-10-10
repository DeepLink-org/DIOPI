/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    AscendTensor inputAt(input), outAt(out);
    if (inputAt.stride() == outAt.stride()) {
        AclOpRunner<1, 1>("Cast", ctx)
            .addInput(inputAt.data(), inputAt.getAclMemBufferSize(), inputAt.getAclMemShape(), inputAt.getAclDataFormat(), inputAt.dtype())
            .addOutput(const_cast<void *>(outAt.data()), outAt.getAclMemBufferSize(), outAt.getAclMemShape(), outAt.getAclDataFormat(), outAt.dtype())
            .setAttr<int32_t>("dst_type", outAt.getAclDataType())
            .run();
    } else {
        diopiTensorHandle_t inputCopy;
        makeTensorLike(ctx, &inputCopy, input, outAt.dtype());
        diopiCastDtype(ctx, inputCopy, input);
        AscendTensor inputCopyAt(inputCopy);

        AclOpRunner<8, 1>("ViewCopy", ctx)
            .addInputWithoutContiguous(out)
            .addConstInput(outAt.shape())
            .addConstInput(outAt.stride())
            .addConstInput(0, diopi_dtype_int64)
            .addInputWithoutContiguous(inputCopy)
            .addConstInput(inputCopyAt.shape())
            .addConstInput(inputCopyAt.stride())
            .addConstInput(0, diopi_dtype_int64)
            .addOutputWithoutContiguous(out)
            .run();
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
