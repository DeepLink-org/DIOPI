/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

std::vector<int> getPermute(diopiConstTensorHandle_t tensorHandle, int64_t dim0, int64_t dim1) {
    DiopiTensor tensor(tensorHandle);
    int inputSize = tensor.shape().size();
    if (tensor.dim() == 0 && (dim0 == 0 || dim0 == -1) && (dim1 == 0 || dim1 == -1)) {
        return std::vector<int>{0};
    }
    int dim0Tmp = static_cast<int>(dim0);
    if (dim0Tmp < 0) {
        dim0Tmp = dim0Tmp + inputSize;
    }

    int dim1Tmp = static_cast<int>(dim1);
    if (dim1Tmp < 0) {
        dim1Tmp = dim1Tmp + inputSize;
    }

    std::vector<int> perms(inputSize);
    for (int i = 0; i < inputSize; i++) {
        perms[i] = i;
    }
    perms[dim0Tmp] = dim1Tmp;
    perms[dim1Tmp] = dim0Tmp;

    return perms;
}

extern "C" {

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> cnnlTransposeDesc;
    cnnlTransposeDescriptor_t transposeDesc = cnnlTransposeDesc.get();
    std::vector<int> perms = getPermute(input, dim0, dim1);
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transposeDesc, perms.size(), perms.data()));

    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    std::vector<DiopiTensor*> inOutTensorVecPtr{&inputTensor, &outputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64, diopi_dtype_int8,
                                           diopi_dtype_bool, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_int64};
    DIOPI_CALL(autoCastTensorType(ctx, inOutTensorVecPtr, supportedDtypes));
    inputTensor = *inOutTensorVecPtr[0];
    outputTensor = *inOutTensorVecPtr[1];

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transposeDesc, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transposeDesc, inputDesc.get(), inputTensor.data(),
                                    outputDesc.get(), outputTensor.data(), workspace, workspaceSize));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
