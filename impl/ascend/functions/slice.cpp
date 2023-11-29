/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t nullOut, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    std::vector<int64_t> index;
    for (int64_t i = start; i < end; i += step) {
        index.push_back(i);
    }
    std::vector<int64_t> dimVec{dim};
    bool negativeIndexSupport = true;
    AclOpRunner<3, 1>("GatherV2", ctx)
        .addInput(input)
        .addConstInput(vectorToDiopiSize(index))
        .addConstInput(vectorToDiopiSize(dimVec))
        .setAttr("negative_index_support", negativeIndexSupport)
        .addOutput(nullOut)
        .run();
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t inputSizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    if (dim < 0) {
        dim += inputSizes.len;
    }
 
    std::vector<int64_t> start64(inputSizes.len, 0);
    std::vector<int64_t> step64(inputSizes.len, 1);
    std::vector<int64_t> end64(inputSizes.data, inputSizes.data + inputSizes.len);
    start64[dim] = start;
    step64[dim] = step;
    end64[dim] = end;
    AclOpRunner<5, 1>("StridedSliceGrad", ctx)
        .addConstInput(inputSizes)
        .addConstInput(vectorToDiopiSize(start64))
        .addConstInput(vectorToDiopiSize(end64))
        .addConstInput(vectorToDiopiSize(step64))
        .addInput(gradOutput)
        .addOutput(gradInput)
        .run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
