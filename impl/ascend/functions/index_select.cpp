/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    std::vector<int64_t> dimVec({dim});
    diopiSize_t dimInput = vectorToDiopiSize(dimVec);
    AclOpRunner<3, 1>("GatherV2", ctx).addInput(input).addInput(index).addConstInput(dimInput).setAttr<int64_t>("batch_dims", 0).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
