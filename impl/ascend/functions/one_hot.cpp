/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {
DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    auto depthScalar = diopiScalar_t();
    depthScalar.stype = diopiDtype_t::diopi_dtype_int64;
    depthScalar.ival = numClasses;
    diopiTensorHandle_t depth;
    makeTensorFromScalar(ctx, &depthScalar, &depth, diopiDtype_t::diopi_dtype_int32);

    printf("numClasses: %ld\n", numClasses);

    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);

    auto onScalar = diopiScalar_t();
    onScalar.stype = diopiDtype_t::diopi_dtype_float64;
    onScalar.fval = 1.0;
    diopiTensorHandle_t on;
    makeTensorFromScalar(ctx, &onScalar, &on, dtype);

    auto offScalar = diopiScalar_t();
    offScalar.stype = diopiDtype_t::diopi_dtype_float64;
    offScalar.fval = 0.0;
    diopiTensorHandle_t off;
    makeTensorFromScalar(ctx, &offScalar, &off, dtype);

    AclOpRunner<4, 1>("OneHot")
        .addInput<0>(input)
        .addConstInput<1>(depth, ACL_FORMAT_ND)
        .addConstInput<2>(on, ACL_FORMAT_ND)
        .addConstInput<3>(off, ACL_FORMAT_ND)
        .setAttr<int>("axis", -1)
        .addOutput(out)
        .run(ctx);
    return diopiSuccess;
}
}
}  // namespace ascend
}  // namespace impl
