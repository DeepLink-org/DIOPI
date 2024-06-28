/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
void calStride(diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat, diopiSize_t* stride) {}

namespace OP_IMPL_NS {

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    BEGIN_CALL_ACL_OP(src, dest);
    if (src == nullptr || dest == nullptr || !srcAt.defined() || !destAt.defined() || srcAt.numel() <= 0 || destAt.numel() <= 0) {
        return diopiSuccess;
    }
    if (at_npu::native::FormatHelper::IsOpInputBaseFormat(destAt) && at_npu::native::FormatHelper::IsOpInputBaseFormat(srcAt)) {
        at_npu::native::NPUNativeFunctions::copy_(destAt, srcAt, false);
    } else {
        at_npu::native::NPUNativeOpApiFunctions::copy_(destAt, srcAt, false);
    }
    END_CALL_ACL_OP();
}
#if 0
diopiError_t diopiContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
    BEGIN_CALL_ACL_OP(input);
    diopiTensorHandle_t result = nullptr;

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiDevice_t device;
    diopiGetTensorDevice(input, &device);

    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);
    at::IntArrayRef atDims(shape.data, shape.len);

    diopiSize_t stride;
    diopiGetTensorStride(input, &stride);

    auto ret = diopiRequireTensor(ctx, &result, &shape, nullptr, dtype, device);
    TORCH_CHECK(diopiSuccess == ret);
    *out = result;

    OP_IMPL_NS::diopiCopyInp(ctx, input, result);
    END_CALL_ACL_OP();
}
#endif

}  // namespace OP_IMPL_NS
