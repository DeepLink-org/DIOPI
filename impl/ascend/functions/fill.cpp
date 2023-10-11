/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <cmath>
#include <limits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *value) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (numel <= 0) {
        return diopiSuccess;
    }

    bool divByZero = true;
    float val = getValue<float>(value);
    if (val == INFINITY) {
        val = 1;
    } else if (val == -INFINITY) {
        val = -1;
    } else if (std::isnan(val)) {
        val = 0;
    } else {
        divByZero = false;
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiTensorHandle_t inputCopy;
    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);

    if (diopi_dtype_bool == dtype && 0 != shape.len) {
        makeTensorLike(ctx, &inputCopy, input, diopi_dtype_int32);
        AclOpRunner<1, 1>("Fills", ctx).addInput(inputCopy).setAttr<float>("value", val).addOutput(inputCopy).run();
        diopiCastDtype(ctx, input, inputCopy);
    } else {
        AclOpRunner<1, 1>("Fills", ctx).addInput(input).setAttr<float>("value", val).addOutput(input).run();
    }
    auto zeroValueScalar = diopiScalar_t();
    zeroValueScalar.stype = diopi_dtype_float64;
    zeroValueScalar.fval = 0.0;

    if (divByZero) diopiDivInpScalar(ctx, input, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
