/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    AclOpRunner<1, 1>("Triu", ctx).addInput(input).setAttr("diagonal", diagonal).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) { return diopiTriu(ctx, input, input, diagonal); }

}  // namespace ascend
}  // namespace impl
