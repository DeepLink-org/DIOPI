/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/diopirt.h>

namespace impl {

namespace ascend_npu {
extern diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
}

namespace ascend {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    return ascend_npu::diopiCastDtype(ctx, out, input);
}

}  // namespace ascend
}  // namespace impl
