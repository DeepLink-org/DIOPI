/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input,
                                 const diopiSize_t splitSizes, int64_t dim) {
    AscendTensor inputTensor(input);

    if (dim < 0) {  // make sure dim is positive
        dim += inputTensor.dim();
    }

    // build the dynamicOutput vector
    std::vector<diopiTensorHandle_t> dynamicOutput;
    for (int64_t i = 0; i < num_outs; i++) {
        AscendTensor outputTensorI(outs[i]);
        std::vector<int64_t> outputShapeI{inputTensor.shape().begin(), inputTensor.shape().end()};
        outputShapeI[dim] = *(splitSizes.data +i);
        makeTensor(ctx, outputTensorI, outputShapeI, inputTensor.dtype());
        dynamicOutput.push_back(const_cast<diopiTensorHandle_t>(outputTensorI.tensorHandle()));
    }

    AclOpRunner<3, 1> ("SplitV", ctx)
        .addInput(input)
        .addConstInput(splitSizes)
        .addConstInput(dim, diopi_dtype_int32)
        .setAttr("num_split", num_outs)
        .addDynamicOutput(dynamicOutput);

    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
