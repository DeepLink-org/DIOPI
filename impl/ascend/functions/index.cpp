/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t zerosLikeInput, diopiConstTensorHandle_t* indices,
                                int64_t nums, diopiConstTensorHandle_t gradOutput) {
    AscendTensor gradInputTensor(gradInput);
    AscendTensor gradOutputTensor(gradOutput);
    if (gradInputTensor.numel() == 0 || gradOutputTensor.numel() == 0) {
        return diopiSuccess;
    }

    std::vector<diopiConstTensorHandle_t> indicesVec;
    indicesVec.reserve(nums);

    for (int i = 0; i < nums; i++) {
        if (indices[i] != nullptr) {
            indicesVec.emplace_back(indices[i]);
        } else {
            int64_t array[1] = {0};
            diopiSize_t size = {array, 1};
            diopiTensorHandle_t emptyTensor = nullptr;
            diopiRequireTensor(ctx, &emptyTensor, &size, nullptr, gradOutputTensor.dtype(), diopi_device);
            indicesVec.emplace_back(emptyTensor);
        }
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, gradInput, zerosLikeInput);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, gradInput, indicesVec, gradOutput, true, false);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
