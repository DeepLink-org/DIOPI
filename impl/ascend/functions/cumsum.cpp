/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    AclOpRunner<2, 1>("Cumsum", ctx).addInput(input).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
