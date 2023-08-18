/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);
    int64_t inputSize = inputShape.getLen();
    if (dim0 < 0) dim0 = dim0 + inputSize;
    if (dim1 < 0) dim1 = dim1 + inputSize;
    std::vector<int64_t> perms(inputSize);
    std::iota(perms.begin(), perms.end(), 0);
    perms[dim0] = dim1;
    perms[dim1] = dim0;
    diopiSize_t permsSize{perms.data(), static_cast<int64_t>(perms.size())};
    AclOpRunner<2, 1>("Transpose", ctx).addInput(input).addConstInput(permsSize).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
