/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    AclOpRunner<1, 1>("Sign", ctx).addInput(input).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiSgn(ctx, input, input); }

}  // namespace ascend
}  // namespace impl
