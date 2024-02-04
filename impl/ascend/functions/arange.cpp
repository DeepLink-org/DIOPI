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
    // convert to integer
    if (isIntegralTypeWithBool(outDtype)) {
        if (!isIntegralTypeWithBool(start->stype)) {
            startTmp = constructDiopiScalarT(outDtype, start->fval);
        } else {
            startTmp.ival = start->ival;
            startTmp->stype = outDtype;
        }
        if (!isIntegralTypeWithBool(end->stype)) {
            endTmp = constructDiopiScalarT(outDtype, end->fval);
        } else {
            endTmp.ival = end->ival;
            endTmp->stype = outDtype;
        }
        if (!isIntegralTypeWithBool(step->stype)) {
            stepTmp = constructDiopiScalarT(outDtype, step->fval);
        } else {
            stepTmp.ival = step->ival;
            stepTmp->stype = outDtype;
        }
    } else {
        if (isIntegralTypeWithBool(start->stype)) {
            startTmp = constructDiopiScalarT(outDtype, start->ival);
        } else {
            startTmp.ival = start->ival;
            startTmp->stype = outDtype;
        }
        if (isIntegralTypeWithBool(end->stype)) {
            endTmp = constructDiopiScalarT(outDtype, end->ival);
        } else {
            endTmp.ival = end->ival;
            endTmp->stype = outDtype;
        }
        if (isIntegralTypeWithBool(step->stype)) {
            stepTmp = constructDiopiScalarT(outDtype, step->ival);
        } else {
            stepTmp.ival = step->ival;
            stepTmp->stype = outDtype;
        }
    }
    AclOpRunner<3, 1>("Range", ctx).addConstInput(startTmp).addConstInput(endTmp).addConstInput(stepTmp).addOutput(out).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
