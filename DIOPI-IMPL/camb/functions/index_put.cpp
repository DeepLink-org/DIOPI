/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor valuesTensor(values);
    DiopiTensor outputTensor(out);

    DIOPI_CHECK(indicesCounts <= inputTensor.dim(), "indices have more indices than self dim");
    DIOPI_CHECK(inputTensor.isContiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(valuesTensor.isContiguous(), "values tensor should be contiguous");
    DIOPI_CHECK(outputTensor.isContiguous(), "output tensor should be contiguous");
    DIOPI_CHECK(inputTensor.dtype() == valuesTensor.dtype(), "input and values must have same dtype");

    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc valuesDesc(valuesTensor, layout);
    CnnlTensorDesc outputDesc(outputTensor, layout);

    // to preserve descriptor and tensor, make sure it's not destructed
    std::vector<CnnlTensorDesc> savedIndicesDescs(indicesCounts);
    std::vector<DiopiTensor> savedIndicesTensors;

    std::vector<cnnlTensorDescriptor_t> indicesDescs;
    std::vector<void*> indicesPtrList;

    bool indicesAllNull = true;
    for (auto i = 0; i < indicesCounts; ++i) {
        DiopiTensor indiceTensor(indices[i]);
        if (indiceTensor.defined()) {
            DIOPI_CHECK(indiceTensor.isContiguous(), "indice tensor should be contiguous");
#if (CNNL_MAJOR <= 1 && CNNL_MINOR < 18)
            DIOPI_CHECK(!(indiceTensor.dtype() == diopi_dtype_bool),
                        "There are bugs in camb kernel when indices dtype is bool, please upgrade your cnnl version to 1.18 at least.");
#endif
            DIOPI_CHECK(indiceTensor.dim() > 0, "zero-dimensional tensor cannot be concatenated");
            DIOPI_CHECK(indiceTensor.dtype() == diopi_dtype_int32 || indiceTensor.dtype() == diopi_dtype_bool || indiceTensor.dtype() == diopi_dtype_uint8,
                        "indiceTensor's dtype should be `int`, `bool` or `uint8`");
            savedIndicesTensors.emplace_back(indiceTensor);
            indicesPtrList.emplace_back(indiceTensor.data());
            savedIndicesDescs[i].set(indiceTensor, layout);
            indicesDescs.emplace_back(savedIndicesDescs[i].get());
            indicesAllNull = false;
        } else {
            indicesPtrList.emplace_back(nullptr);
            indicesDescs.emplace_back(nullptr);
        }
    }
    if (indicesAllNull) {
        // cnnl can't support all of the indices are nullptr
        return diopiSuccess;
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(
        cnnlGetIndexPutWorkspaceSize(handle, inputDesc.get(), indicesDescs.data(), indicesDescs.size(), valuesDesc.get(), accumulate, &workspaceSize));

    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlIndexPut(handle,
                                inputDesc.get(),
                                inputTensor.data(),
                                indicesDescs.data(),
                                indicesPtrList.data(),
                                indicesDescs.size(),
                                valuesDesc.get(),
                                valuesTensor.data(),
                                workspacePtr,
                                workspaceSize,
                                accumulate,
                                true,
                                outputDesc.get(),
                                outputTensor.data()));

    return diopiSuccess;
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    DIOPI_CALL(diopiIndexPut(ctx, input, input, values, indices, indicesCounts, accumulate));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
