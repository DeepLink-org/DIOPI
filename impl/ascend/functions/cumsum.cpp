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
DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    AclOpRunner<2, 1>("Cumsum", ctx).addInput(input).addConstInput<int64_t>(dim).addOutput(out).run();
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl