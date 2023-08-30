/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end,
                                              const diopiScalar_t* step) {
    AclOpRunner<3, 1>("Range", ctx).addConstInput(*start).addConstInput(*end).addConstInput(*step).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
