/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    std::vector<int64_t> index;
    for (int64_t i = start; i < end; i += step) {
        index.push_back(i);
    }
    std::vector<int64_t> dim_vec{dim};
    bool negative_index_support = true;
    AclOpRunner<3, 1>("GatherV2", ctx)
        .addInput(input)
        .addConstInput(vectorToDiopiSize(index))
        .addConstInput(vectorToDiopiSize(dim_vec))
        .setAttr("negative_index_support", negative_index_support)
        .addOutput(null_out)
        .run();
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    if (dim < 0) {
        dim += input_sizes.len;
    }
    std::vector<int64_t> index;
    for (int64_t i = start; i < end; i += step) {
        index.push_back(i);
    }

    std::vector<int64_t> start64(input_sizes.len, 0);
    std::vector<int64_t> step64(input_sizes.len, 1);
    std::vector<int64_t> end64(input_sizes.data, input_sizes.data + input_sizes.len);
    start64[dim] = start;
    step64[dim] = step;
    end64[dim] = end;
    AclOpRunner<5, 1>("StridedSliceGrad", ctx)
        .addConstInput(input_sizes)
        .addConstInput(vectorToDiopiSize(start64))
        .addConstInput(vectorToDiopiSize(end64))
        .addConstInput(vectorToDiopiSize(step64))
        .addInput(grad_output)
        .addOutput(grad_input)
        .run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
