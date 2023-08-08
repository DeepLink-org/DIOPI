/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" {
DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                                      int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    std::vector<int64_t> dimVec({0});
    diopiSize_t dim(dimVec.data(), dimVec.size());
    AclOpRunner<3, 1>("GatherV2", ctx).addInput(weight, indices).addConstInput(dim).setAttr<int64_t>("batch_dims", 0).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                              diopiConstTensorHandle_t indices, int64_t numWeights, int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    diopiTensorHandle_t indicesInt32;
    makeTensorLike(ctx, &indicesInt32, indices, diopi_dtype_int32);
    diopiCastDtype(ctx, indicesInt32, indices);
    AclOpRunner<2, 1>("EmbeddingDenseGrad", ctx)
        .addInput(grad, indicesInt32)
        .setAttr("num_weights", numWeights)
        .setAttr("padding_idx", paddingIdx)
        .setAttr("scale_grad_by_freq", scaleGradByfreq)
        .addOutput(out)
        .run();
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
