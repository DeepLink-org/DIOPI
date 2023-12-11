/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    AclOpRunner<3, 1>("LinSpace", ctx)
        .addConstInput(*start, diopi_dtype_float32)
        .addConstInput(*end, diopi_dtype_float32)
        .addConstInput(steps, diopi_dtype_int32)
        .addOutput(out)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
