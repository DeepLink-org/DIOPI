/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

std::vector<int64_t> nonzeroNpuMaxOutputSize(diopiConstTensorHandle_t input) {
    int64_t inputNumEl;
    diopiGetTensorNumel(input, &inputNumEl);
    diopiSize_t inputSize;
    diopiGetTensorShape(input, &inputSize);
    int64_t inputDim = inputSize.len;
    std::vector<int64_t> maxOutputSize({inputNumEl, inputDim});
    return maxOutputSize;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    auto outputSizeVec = nonzeroNpuMaxOutputSize(input);
    diopiSize_t outputSize = vectorToDiopiSize(outputSizeVec);

    diopiTensorHandle_t output;
    diopiRequireTensor(ctx, &output, &outputSize, nullptr, diopi_dtype_int64, diopi_device);

    AclOpRunner<1, 1>("NonZero", ctx).addInput(input).setAttr("transpose", false).addSyncOutput(&output, ACL_FORMAT_NCHW).run();
    *out = output;
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
