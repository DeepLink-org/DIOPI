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
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);

    DIOPI_CALL(dataTypeCast(ctx, outputTr, inputTr));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
