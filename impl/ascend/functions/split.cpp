/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {
diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t numOuts, diopiConstTensorHandle_t input,
                                 const diopiSize_t splitSizes, int64_t dim) {
    AscendTensor inputTensor(input);

    if (dim < 0) {  // make sure dim is positive
        dim += inputTensor.dim();
    }

    // build the dynamicOutput vector
    std::vector<diopiTensorHandle_t> dynamicOutput;

    for (int64_t i = 0; i < numOuts; i++) {
        dynamicOutput.push_back(outs[i]);
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnSplitWithSize, ctx, input, splitSizes, dim, dynamicOutput);

    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
