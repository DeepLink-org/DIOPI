/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t *indices, int64_t nums) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    int64_t arraySize = 8;
    DiopiTensor inputTensor(input);
    DIOPI_CHECK(inputTensor.dim() <= 8, "The dimension of input tensor cannot be larger than 8");
    DIOPI_CHECK(nums <= inputTensor.dim(), "Too many indices for tensor of dimension");
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    std::vector<DiopiTensor*> tensor{&inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensor, supportedDtypes));

    std::vector<CnnlTensorDesc> indicesTensors(arraySize);
    std::vector<CnnlTensorDesc> indicesDesc(arraySize);
    std::vector<cnnlTensorDescriptor_t> indicesDescT(arraySize);
    std::vector<void *> indicesPtrList(arraySize);

    for (int64_t i = 0; i < arraySize; i++) {
        if (i < nums) {
            DiopiTensor indiceTensor(indices[i]);
            if (indiceTensor.defined()) {
                DIOPI_CHECK(indiceTensor.dim() <= 8, "The dimension of indice tensor cannot be larger than 8");
                DIOPI_CALL(autoCastTensorType(ctx, {&indiceTensor}, {diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_int32}));
                printDevData(ctx, indiceTensor, "indice");
                indicesDesc[i].set(indiceTensor, CNNL_LAYOUT_ARRAY);
                indicesDescT[i] = indicesDesc[i].get();
                indicesPtrList[i] = indiceTensor.data();
            } else {
                indicesDescT[i] = nullptr;
                indicesPtrList[i] = nullptr;
            }
        } else {
            indicesDescT[i] = nullptr;
            indicesPtrList[i] = nullptr;
        }
    }

   
    int outputDescDim = 0;
    std::vector<int> outputDescDims(arraySize);
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexOutputDim(handle, inputDesc.get(), indicesDescT.data(), &outputDescDim, outputDescDims.data()));
    std::cout <<"outputDescDim: "<<outputDescDim << std::endl;
    for (auto num : outputDescDims) {
        std::cout << num << " ";
    }
    std::cout<<std::endl;
    outputDescDims.resize(outputDescDim);
    std::cout <<"outputDescDim After: "<<outputDescDim << std::endl;
    for (auto num : outputDescDims) {
        std::cout << num << " ";
    }
    std::cout<<std::endl;

    std::vector<int64_t> outTensorShape(outputDescDims.begin(), outputDescDims.end());
    DiopiTensor outTensor = requiresTensor(ctx, outTensorShape, inputTensor.dtype());
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexWorkspaceSize(handle, inputDesc.get(), indicesDescT.data(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlAdvancedIndex(handle,
                                     inputDesc.get(),
                                     inputTensor.data(),
                                     indicesDescT.data(),
                                     indicesPtrList.data(),
                                     workspace,
                                     workspaceSize,
                                     outDesc.get(),
                                     outTensor.data(),
                                     &outputDescDim,
                                     outputDescDims.data()));
    *out = diopiTensorHandle_t(outTensor);
    return diopiSuccess;
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                diopiConstTensorHandle_t *indices, int64_t nums, diopiConstTensorHandle_t grad) {
    DIOPI_CALL(diopiIndexPut(ctx, grad_input, zeros_like_input, grad, indices, nums, true));
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
