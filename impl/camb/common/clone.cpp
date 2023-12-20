/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"
#include "debug.hpp"

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
    diopiMemoryFormat_t destMemoryFormat = 
    if (inTensor.shape() == outTensor.shape() && inTensor.dim() != 0 && inTensor.dtype() != diopi_dtype_float64 && inTensor.dtype() == outTensor.dtype() &&
        denseCheck(inTensor) && denseCheck(outTensor) && (outTensor.isContiguous() || inTensor.isContiguous())) {
        DIOPI_CALL(contiguousOut(ctx, const_cast<DiopiTensor&>(inTensor), outTensor, outTensor.isContiguous()));
        return diopiSuccess;
    }
    CnnlTensorDesc inTensorDesc(inTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTensorDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALL_CNNL(cnnlCopy(handle, inTensorDesc.get(), inTensor.data(), outTensorDesc.get(), outTensor.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
