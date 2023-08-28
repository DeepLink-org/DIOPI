/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    for (int i = 0; i < inputsNum; i++) {
        DiopiTensor inputTensor(inputs[i]);
        DiopiTensor outTensor(outs[i]);

        auto inputDim = inputTensor.shape();
        auto outputDims = outTensor.shape();

        int tmpOutputDims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int tmpInputDims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int repeatDim0 = 1;
        int repeatDim1 = 1;
        for (int j = 0; j < i; j++) {
            repeatDim0 *= outputDims[j];
        }
        for (int k = i + 1; k < inputsNum; k++) {
            repeatDim1 *= outputDims[k];
        }
        tmpOutputDims[0] = repeatDim0 * outputDims[i];
        tmpOutputDims[1] = repeatDim1;
        tmpInputDims[0] = outputDims[i];
        tmpInputDims[1] = 1;

        CnnlTensorDesc inputDesc;
        CnnlTensorDesc outDesc;
        std::vector<int> inDims = {tmpInputDims[0], tmpInputDims[1]};
        std::vector<int> outDims = {tmpOutputDims[0], tmpOutputDims[1]};
        inputDesc.set(inputTensor, CNNL_LAYOUT_ARRAY, inDims);
        outDesc.set(outTensor, CNNL_LAYOUT_ARRAY, outDims);

        DIOPI_CALLCNNL(cnnlTile(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
