/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AscendTensor inputAt(input), outAt(out);
    AclOpRunner<1, 1>("Cast", ctx)
        .addInput(inputAt.data(), inputAt.getAclMemBufferSize(), inputAt.getAclMemShape(), inputAt.getAclDataFormat(), inputAt.dtype())
        .addOutput(const_cast<void *>(outAt.data()), outAt.getAclMemBufferSize(), outAt.getAclMemShape(), outAt.getAclDataFormat(), outAt.dtype())
        .setAttr<int32_t>("dst_type", outAt.getAclDataType())
        .run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
