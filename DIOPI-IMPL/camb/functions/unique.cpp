#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, const int64_t *dim, bool sorted, bool returnCounts,
                         diopiTensorHandle_t indices, diopiTensorHandle_t *counts) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // input_tensor
    DiopiTensor inputTensor(input);
    int realDim = -1;  // If dim is set to -1, the unique of the flattened input is to apply in CNNL.
    if (dim != nullptr) {
        realDim = ((*dim) < 0) ? (*dim + inputTensor.dim()) : *dim;
    }

    // dtype cast
    diopiDtype_t originInputDtype = inputTensor.dtype();
    std::vector<DiopiTensor *> pTensors{&inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // output_tensor
    DiopiTensor outputTensor =
        (realDim != -1) ? requiresTensor(ctx, {inputTensor.shape()}, inputTensor.dtype()) : requiresTensor(ctx, {inputTensor.numel()}, inputTensor.dtype());
    // index_tensor
    DiopiTensor indexTensor = (realDim != -1) ? requiresTensor(ctx, {inputTensor.shape()[realDim]}, diopi_dtype_int32)
                                              : requiresTensor(ctx, {inputTensor.numel()}, diopi_dtype_int32);
    // counts_tensor
    DiopiTensor countsTensor = (realDim != -1) ? requiresTensor(ctx, {outputTensor.shape()[realDim]}, diopi_dtype_int32)
                                               : requiresTensor(ctx, {outputTensor.numel()}, diopi_dtype_int32);

    // Tensor Desc
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc countsDesc(countsTensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlUniqueDescriptor_t, cnnlCreateUniqueDescriptor, cnnlDestroyUniqueDescriptor> uniqueDesc;

    // torch.unique always sort the tensor at the beginning
    // regardless of the sort argument when dim is specified
    if (*dim != -1) {
        sorted = true;
    }
    cnnlUniqueSort_t mode = sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_FORWARD;
    bool returnIndices = (indices != nullptr) ? true : false;

    if (mode == CNNL_UNSORT_FORWARD) {
        DIOPI_CHECK((inputTensor.dim() == 1),
                    "the dimension of input must be one-dimensional "
                    "when mode is CNNL_UNSORT_FORWARD");
    }
    DIOPI_CALLCNNL(cnnlSetUniqueDescriptor(uniqueDesc.get(), mode, realDim, returnIndices, returnCounts));
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetUniqueWorkspaceSize(handle, uniqueDesc.get(), inputDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    std::vector<int64_t> temp{1, 1};
    DiopiTensor outlenTensor = requiresTensor(ctx, temp, diopi_dtype_int32);

    DIOPI_CALLCNNL(cnnlUnique_v2(handle,
                                 uniqueDesc.get(),
                                 inputDesc.get(),
                                 inputTensor.data(),
                                 workspace,
                                 workspaceSize,
                                 static_cast<int *>(outlenTensor.data()),
                                 outputDesc.get(),
                                 outputTensor.data(),
                                 indexDesc.get(),
                                 returnIndices ? indexTensor.data() : nullptr,
                                 countsDesc.get(),
                                 returnCounts ? countsTensor.data() : nullptr));

    DIOPI_CALL(dataTypeCast(ctx, outputTensor, originInputDtype));
    int32_t outlenHost = 0;
    cnrtMemcpyAsync(&outlenHost, outlenTensor.data(), sizeof(int32_t), getStream(ctx), cnrtMemcpyDevToHost);
    cnrtQueueSync(getStream(ctx));

    std::vector<int64_t> trueOutShape = inputTensor.shape();
    if (realDim != -1) {
        trueOutShape[realDim] = outlenHost;
    } else {
        trueOutShape[0] = outlenHost;
    }
    DiopiTensor slicedOutputTensor = requiresTensor(ctx, trueOutShape, outputTensor.dtype());
    DiopiTensor slicedCountsTensor = requiresTensor(ctx, {outlenHost}, diopi_dtype_int32);

    diopiSlice(ctx, diopiTensorHandle_t(slicedOutputTensor), diopiTensorHandle_t(outputTensor), realDim, 0, outlenHost, 1);
    diopiSlice(ctx, diopiTensorHandle_t(slicedCountsTensor), diopiTensorHandle_t(countsTensor), 0, 0, outlenHost, 1);

    *out = diopiTensorHandle_t(slicedOutputTensor);
    if (returnIndices) {
        DiopiTensor trueIndexTensor(indices);
        CnnlTensorDesc trueIndexDesc(trueIndexTensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALL(dataTypeCast(ctx, indexTensor, diopi_dtype_int64));
        CnnlTensorDesc indexDesc64(indexTensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCopy(handle, indexDesc64.get(), indexTensor.data(), trueIndexDesc.get(), trueIndexTensor.data()));
    }
    if (returnCounts) {
        *counts = diopiTensorHandle_t(slicedCountsTensor);
    }
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
