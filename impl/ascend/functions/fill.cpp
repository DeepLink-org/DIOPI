/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cfloat>
#include <cmath>
#include <limits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *value) {
    float val = getValue<float>(value);

    bool divByZero = true;

    if (val == INFINITY) {
        val = 1;
    } else if (val == -INFINITY) {
        val = -1;
    } else if (val == NAN) {
        val = 0;
    } else {
        divByZero = false;
    }

    auto zeroValueScalar = diopiScalar_t();
    zeroValueScalar.stype = diopi_dtype_float64;
    zeroValueScalar.fval = 0.0;
    AclOpRunner<1, 1>("Fills", ctx).addInput(input).setAttr<float>("value", val).addOutput(input).run();
    if (divByZero) diopiDivInpScalar(ctx, input, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
