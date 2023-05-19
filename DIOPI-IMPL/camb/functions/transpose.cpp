/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

std::vector<int> getPerm(diopiConstTensorHandle_t tensorHandle,
                         int64_t dim0,
                         int64_t dim1) {
    DiopiTensor tensor(tensorHandle);
    int inputSize = tensor.shape().size();

    int dim0Tmp = static_cast<int>(dim0);
    if (dim0Tmp < 0) {
        dim0Tmp = dim0Tmp + inputSize;
    }

    int dim1Tmp = static_cast<int>(dim1);
    if (dim1Tmp < 0) {
        dim1Tmp = dim1Tmp + inputSize;
    }

    std::vector<int> perms(inputSize);
    std::iota(perms.begin(), perms.end(), 0);

    perms[dim0Tmp] = dim1Tmp;
    perms[dim1Tmp] = dim0Tmp;

    return perms;
}

extern "C" {

diopiError_t diopiTranspose(diopiContextHandle_t ctx,
                                      diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input,
                                      int64_t dim0,
                                      int64_t dim1) {
    auto stream = getStream(ctx);
    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> cnnlHandle;
    cnnlHandle_t handle = cnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTransposeDescriptor_t,
                      cnnlCreateTransposeDescriptor,
                      cnnlDestroyTransposeDescriptor>
        cnnlTransposeDesc;
    cnnlTransposeDescriptor_t transposeDesc = cnnlTransposeDesc.get();
    std::vector<int> perms = getPerm(input, dim0, dim1);
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transposeDesc, perms.size(), perms.data()));

    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);
    const void* inputPtr = inputTensor.data();
    void* outPtr = outputTensor.data();

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(
        handle, inputDesc.get(), transposeDesc, &workspaceSize));
    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle,
                                    transposeDesc,
                                    inputDesc.get(),
                                    inputPtr,
                                    outputDesc.get(),
                                    outPtr,
                                    workspace,
                                    workspaceSize));
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
