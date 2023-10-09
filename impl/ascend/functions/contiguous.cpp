/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <climits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
    *out = contiguous(ctx, input, memoryFormat);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
