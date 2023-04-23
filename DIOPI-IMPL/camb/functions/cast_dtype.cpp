/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor input_tr(input);
    DiopiTensor output_tr(out);

    DIOPI_CALL(dataTypeCast(ctx, output_tr, input_tr));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
