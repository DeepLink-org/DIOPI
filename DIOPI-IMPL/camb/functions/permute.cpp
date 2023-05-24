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

extern "C" diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    if (diopi_dtype_float64 == inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);

    std::vector<int> permData{dims.data, dims.data + dims.len};
    for (int& i : permData) {
        if (i < 0) {
            i += inputTensor.dim();
        }
    }

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> transDesc;

    const std::vector<int64_t> srcInputShape = inputTensor.shape();
    int dimNum = srcInputShape.size();
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transDesc.get(), dimNum, permData.data()));
    size_t workspaceSize;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    if (inputTensor.dtype() == outputTensor.dtype()) {
        DIOPI_CALLCNNL(
            cnnlTranspose_v2(handle, transDesc.get(), inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTensor.data(), workspace, workspaceSize));
    } else {
        DiopiTensor outTemp = requiresTensor(ctx, outputTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlTranspose_v2(handle, transDesc.get(), inputDesc.get(), inputTensor.data(), outTempDesc.get(), outTemp.data(), workspace, workspaceSize));
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTemp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
