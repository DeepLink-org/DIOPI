/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

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
        AscendTensor outputTensorI(outs[i]);
        dynamicOutput.push_back(const_cast<diopiTensorHandle_t>(outs[i]));
    }

    AclOpRunner<3, 1>("SplitV", ctx)
        .addInput(input)
        .addConstInput(splitSizes)
        .addConstInput(dim, diopi_dtype_int64)
        .setAttr("num_split", numOuts)
        .addDynamicOutput(dynamicOutput, inputTensor.dtype())
        .run();

    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
