/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    diopiScalar_t startTmp;
    diopiScalar_t endTmp;
    diopiScalar_t stepTmp;
    diopiDtype_t outDtype;

    diopiGetTensorDtype(out, &outDtype);
    startTmp.stype = outDtype;
    endTmp.stype = outDtype;
    stepTmp.stype = outDtype;

    // convert to integer
    if (isIntegralTypeWithBool(outDtype)) {
        if (!isIntegralTypeWithBool(start->stype)) {
            startTmp.ival = static_cast<int64_t>(start->fval);
        } else {
            startTmp.ival = start->ival;
        }
        if (!isIntegralTypeWithBool(end->stype)) {
            endTmp.ival = static_cast<int64_t>(end->fval);
        } else {
            endTmp.ival = end->ival;
        }
        if (!isIntegralTypeWithBool(step->stype)) {
            stepTmp.ival = static_cast<int64_t>(step->fval);
        } else {
            stepTmp.ival = step->ival;
        }
    } else {
        if (isIntegralTypeWithBool(start->stype)) {
            startTmp.fval = static_cast<double>(start->ival);
        } else {
            startTmp.ival = start->ival;
        }
        if (isIntegralTypeWithBool(end->stype)) {
            endTmp.fval = static_cast<double>(end->ival);
        } else {
            endTmp.ival = end->ival;
        }
        if (isIntegralTypeWithBool(step->stype)) {
            stepTmp.fval = static_cast<double>(step->ival);
        } else {
            stepTmp.ival = step->ival;
        }
    }
    AclOpRunner<3, 1>("Range", ctx).addConstInput(startTmp).addConstInput(endTmp).addConstInput(stepTmp).addOutput(out).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
