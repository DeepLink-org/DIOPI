

#include <algorithm>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor = DiopiTensor(input);
    DiopiTensor indicesTensor = DiopiTensor(out);

    DiopiTensor inputCasted = inputTensor;
    DiopiTensor indicesCasted = indicesTensor;

    DIOPI_CALL(autoCastTensorType(ctx, {&inputCasted}, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32}));
    DIOPI_CALL(autoCastTensorType(ctx, {&indicesCasted}, {diopi_dtype_int32}));

    DiopiTensor outCasted = requiresTensor(ctx, indicesCasted.shape(), inputCasted.dtype());

    std::vector<int> axis;
    if (dim == nullptr) {
        for (int i = 0; i < inputCasted.shape().size(); ++i) {
            axis.push_back(i);
        }
    } else {
        axis.push_back((*dim < 0) ? (*dim + inputCasted.shape().size()) : *dim);
    }

    std::vector<int> inputDims, outputDims;
    if (axis.size() > 1) {
        for (int i = 0; i < inputCasted.shape().size(); ++i) {
            int dim = inputCasted.shape()[i];
            if (std::find(axis.begin(), axis.end(), i) != axis.end()) {  // found
                if (inputDims.empty()) {
                    inputDims.push_back(dim);
                    outputDims.push_back(1);
                } else {
                    inputDims[inputDims.size() - 1] *= dim;
                }
            } else {
                inputDims.push_back(dim);
                outputDims.push_back(dim);
            }
        }
    } else {
        for (int dim : inputCasted.shape()) {
            inputDims.push_back(dim);
        }
        for (int dim : outCasted.shape()) {
            outputDims.push_back(dim);
        }
    }
    axis.resize(1);
    CnnlTensorDesc inputDesc(inputCasted, CNNL_LAYOUT_ARRAY, inputDims);
    CnnlTensorDesc outDesc(outCasted, CNNL_LAYOUT_ARRAY, outputDims);

    cnnlDataType_t tensorType;
    CnnlDataType::convertToCnnlType(&tensorType, inputCasted.dtype());
    CnnlResourceGuard<cnnlReduceDescriptor_t, cnnlCreateReduceDescriptor, cnnlDestroyReduceDescriptor> reduceDesc;

    DIOPI_CALLCNNL(cnnlSetReduceDescriptor_v2(
        reduceDesc.get(), axis.data(), axis.size(), CNNL_REDUCE_MAX, tensorType, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_ONLY_INDICES, CNNL_32BIT_INDICES, 0.0));

    void* workspace = nullptr;
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, inputDesc.get(), outDesc.get(), reduceDesc.get(), &workspaceSize));
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlReduce(handle,
                              reduceDesc.get(),
                              workspace,
                              workspaceSize,
                              nullptr,
                              inputDesc.get(),
                              inputCasted.data(),
                              indicesCasted.elemsize() * indicesCasted.numel(),
                              indicesCasted.data(),
                              nullptr,
                              outDesc.get(),
                              outCasted.data()));

    DIOPI_CALL(dataTypeCast(ctx, indicesTensor, indicesCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
