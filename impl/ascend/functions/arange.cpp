/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    diopiScalar_t startTemp;
    diopiScalar_t endTemp;
    diopiScalar_t stepTemp;

    // The pre-processing ensures that the data types of start, end, and step are consistent.
    if (!((isIntegralTypeWithBool(start->stype) == isIntegralTypeWithBool(end->stype)) &&
          (isIntegralTypeWithBool(start->stype) == isIntegralTypeWithBool(step->stype)))) {
        if (isIntegralTypeWithBool(start->stype)) {
            startTemp.stype = diopi_dtype_float64;
            startTemp.fval = static_cast<double>(start->ival);
        } else {
            startTemp = *start;
        }

        if (isIntegralTypeWithBool(end->stype)) {
            endTemp.stype = diopi_dtype_float64;
            endTemp.fval = static_cast<double>(end->ival);
        } else {
            endTemp = *end;
        }

        if (isIntegralTypeWithBool(step->stype)) {
            stepTemp.stype = diopi_dtype_float64;
            stepTemp.fval = static_cast<double>(step->ival);
        } else {
            stepTemp = *step;
        }

    } else {
        startTemp = *start;
        endTemp = *end;
        stepTemp = *step;
    }
    AclOpRunner<3, 1>("Range", ctx).addConstInput(startTemp).addConstInput(endTemp).addConstInput(stepTemp).addOutput(out).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
