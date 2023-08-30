/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

diopiError_t clone(diopiContextHandle_t ctx, const DiopiTensor& inTensor, DiopiTensor& outTensor, diopiMemoryFormat_t memoryFormat) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    if (memoryFormat == diopiMemoryFormat_t::Preserve) {
        // torch.preserve_format: Used in functions like clone to preserve the memory format of the input tensor.
        // If input tensor is allocated in dense non-overlapping memory, the output tensor strides will be copied from the input.
        // Otherwise output strides will follow torch.contiguous_format.
        // Based on the above, we should check further whether the cnnlCopy take overlapping memory into account
        outTensor = requiresTensor(ctx, inTensor.shape(), inTensor.stride(), inTensor.dtype());
    } else {
        outTensor = requiresTensor(ctx, inTensor.shape(), inTensor.dtype(), memoryFormat);
    }
    CnnlTensorDesc inTensorDesc(inTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTensorDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlCopy(handle, inTensorDesc.get(), inTensor.data(), outTensorDesc.get(), outTensor.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
