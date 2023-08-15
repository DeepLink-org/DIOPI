/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    AclOpRunner<2, 1>("ArgMaxV2", ctx).addInput(input).addConstInput(*dim).setAttr("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
