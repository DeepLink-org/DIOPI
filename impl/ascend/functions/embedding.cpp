/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                            int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    std::vector<int64_t> dimVec({0});
    diopiSize_t dim = vectorToDiopiSize(dimVec);
    AclOpRunner<3, 1>("GatherV2", ctx).addInput(weight).addInput(indices).addConstInput(dim).setAttr<int64_t>("batch_dims", 0).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                    int64_t numWeights, int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    AclOpRunner<2, 1>("EmbeddingDenseGrad", ctx)
        .addInput(grad)
        .addInput(indices, diopi_dtype_int32)
        .setAttr("num_weights", numWeights)
        .setAttr("padding_idx", paddingIdx)
        .setAttr("scale_grad_by_freq", scaleGradByfreq)
        .addOutput(out)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
