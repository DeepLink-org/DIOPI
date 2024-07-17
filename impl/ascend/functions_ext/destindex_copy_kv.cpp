/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"
#include "../error.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    AscendTensor destLocAt(destLoc);
    diopiSize_t destLocSize;
    diopiGetTensorShape(destLoc, &destLocSize);

    if (destLocSize.len != 1) {
        setLastErrorString("only support destLoc.rank == 1");
        return diopiNoImplement;
    }

    std::vector<int64_t> shape(destLocAt.shape());
    shape.push_back(1);

    auto destLocReshape = destLocAt.view(shape);

    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterNd, ctx, out, destLocReshape, k, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
