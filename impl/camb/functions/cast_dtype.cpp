/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);

    DIOPI_CALL(dataTypeCast(ctx, outputTr, inputTr));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
