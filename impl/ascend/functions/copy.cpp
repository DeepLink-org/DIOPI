/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/diopirt.h>

namespace impl {

namespace ascend_npu {
extern diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest);
}  // namespace ascend_npu

namespace ascend {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) { return ascend_npu::diopiCopyInp(ctx, src, dest); }

}  // namespace ascend
}  // namespace impl
