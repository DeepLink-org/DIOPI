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

extern "C" {
/**
 * @brief Returns the standard derivation of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16].
 * @param dim an array, dimension for reduction. type = [int32, int64].
 * @param unbiased whether to compute the unbiased standard deviation.
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // shape of outTensor does not keep the dim.
    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    auto outDtype = outTensor.dtype();

    bool keepDim = false;
    if (outTensor.dim() == inputTensor.dim()) {
        keepDim = true;
    }

    int axisNum = 0;
    int *axis = nullptr;
    if (dim.getLen() == 0) {
        axisNum = inputTensor.dim();
        axis = new int[axisNum];
        for (int i = 0; i < axisNum; i++) {
            axis[i] = i;
        }
    } else {
        axisNum = dim.getLen();
        axis = new int[axisNum];
        for (int i = 0; i < axisNum; i++) {
            axis[i] = dim.data[i];
        }
    }

    if (!keepDim) {
        auto outKeepDimShape = inputTensor.shape();
        for (int i = 0; i < axisNum; i++) {
            outKeepDimShape[axis[i]] = 1;
        }
        auto outKeepDimStride = inputTensor.stride();
        for (int i = inputTensor.dim() - 2; i >= 0; i--) {
            outKeepDimStride[i] = outKeepDimStride[i+1] * outKeepDimShape[i+1];
        }
        outTensor.asStrided(outKeepDimShape, outKeepDimStride);
    }

    // cast supported dtyeps for tensors.
    std::vector<DiopiTensor *> tensorVecPtr{&outTensor, &inputTensor};
    std::set<diopiDtype_t> supportedDtype{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, tensorVecPtr, supportedDtype));
    outTensor = *tensorVecPtr[0];
    inputTensor = *tensorVecPtr[1];

    // tensor descriptor for cnnl.
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlStdVarMeanDescriptor_t, cnnlCreateStdVarMeanDescriptor, cnnlDestroyStdVarMeanDescriptor> stdVarMeanObj;
    cnnlStdVarMeanDescriptor_t stdVarMeanDesc = stdVarMeanObj.get();
    DIOPI_CALLCNNL(cnnlSetStdVarMeanDescriptor(stdVarMeanDesc, CNNL_STD, axisNum, axis, unbiased));
    delete [] axis;

    size_t workspaceSize = 0;
    void *workspace = nullptr;
    DIOPI_CALLCNNL(cnnlGetStdVarMeanWorkspaceSize(handle, stdVarMeanDesc, inputDesc.get(), &workspaceSize));
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlStdVarMean(handle,
                                  stdVarMeanDesc,
                                  inputDesc.get(),
                                  inputTensor.data(),
                                  workspace,
                                  workspaceSize,
                                  outDesc.get(),
                                  outTensor.data(),
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr));
    if (!keepDim) {
        std::vector<int64_t> outShape;
        for (int i = 0; i < outTensor.dim(); i++) {
            if (outTensor.shape()[i] > 1) {
                outShape.push_back(outTensor.shape()[i]);
            }
        }
        std::vector<int64_t> outStride(outShape.size(), 1);
        for(int i = outShape.size() - 2; i >= 0; i--) {
            outStride[i] = outStride[i+1] * outShape[i+1];
        }
        outTensor.asStrided(outShape, outStride);
    }
    if (outTensor.dtype() != outDtype) {
        dataTypeCast(ctx, outTensor, outDtype);
    }

    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
