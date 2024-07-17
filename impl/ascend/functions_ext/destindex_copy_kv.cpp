/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include <vector>

#include "../aclnn/adaptor.hpp"
#include "../error.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    AscendTensor destLocAt(destLoc);
    std::vector<int64_t> destLocShape = destLocAt.shape();

    if (destLocShape.size() != 1) {
        setLastErrorString("only support destLoc.rank == 1");
        return diopiNoImplement;
    }

    std::vector<int64_t> shape(destLocShape);
    shape.emplace_back(1);

    auto destLocReshape = destLocAt.view(shape);

    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterNd, ctx, out, destLocReshape, k, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
