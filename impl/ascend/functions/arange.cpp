/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    diopiScalar_t* startTmp = const_cast<diopiScalar_t*>(start);
    diopiScalar_t* endTmp = const_cast<diopiScalar_t*>(end);
    diopiScalar_t* stepTmp = const_cast<diopiScalar_t*>(step);
    diopiDtype_t outDtype;
    diopiGetTensorDtype(out, &outDtype);
    // convert to integer
    if (isIntegralTypeWithBool(outDtype)) {
        if (!isIntegralTypeWithBool(start->stype)) {
            *startTmp = constructDiopiScalarT(outDtype, start->fval);
        } else {
            startTmp->stype = outDtype;
        }
        if (!isIntegralTypeWithBool(end->stype)) {
            *endTmp = constructDiopiScalarT(outDtype, end->fval);
        } else {
            endTmp->stype = outDtype;
        }
        if (!isIntegralTypeWithBool(step->stype)) {
            *stepTmp = constructDiopiScalarT(outDtype, step->fval);
        } else {
            stepTmp->stype = outDtype;
        }
    } else {
        if (isIntegralTypeWithBool(start->stype)) {
            *startTmp = constructDiopiScalarT(outDtype, start->ival);
        } else {
            startTmp->stype = outDtype;
        }
        if (isIntegralTypeWithBool(end->stype)) {
            *endTmp = constructDiopiScalarT(outDtype, end->ival);
        } else {
            endTmp->stype = outDtype;
        }
        if (isIntegralTypeWithBool(step->stype)) {
            *stepTmp = constructDiopiScalarT(outDtype, step->ival);
        } else {
            stepTmp->stype = outDtype;
        }
    }
    AclOpRunner<3, 1>("Range", ctx).addConstInput(*startTmp).addConstInput(*endTmp).addConstInput(*stepTmp).addOutput(out).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
