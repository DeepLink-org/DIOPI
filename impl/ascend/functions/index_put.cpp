/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    AscendTensor inputTensor(input);
    AscendTensor valuesTensor(values);
    if (inputTensor.numel() == 0 || valuesTensor.numel() == 0) {
        return diopiSuccess;
    }

    std::vector<diopiConstTensorHandle_t> indicesVec;
    indicesVec.reserve(indicesCounts);

    for (int i = 0; i < indicesCounts; i++) {
        if (indices[i] != nullptr) {
            indicesVec[i] = indices[i];
        } else {
            int64_t array[1] = {0};
            diopiSize_t size = {array, 1};
            diopiTensorHandle_t emptyTensor = nullptr;
            diopiRequireTensor(ctx, &emptyTensor, &size, nullptr, inputTensor.dtype(), diopi_device);
            indicesVec[i] = emptyTensor;
        }
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, input, indicesVec, values, accumulate, false);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
