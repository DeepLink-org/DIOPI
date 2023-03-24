#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

namespace {
aclDataType dtypeConvertor(diopiConstTensorHandle_t th) {
    auto dtype = getAclDataType(th);
    if (dtype == ACL_BOOL) {
        return ACL_UINT8;
    }
    return dtype;
}

}  // namespace

extern "C" DIOPI_API diopiError_t
diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    const float value = (alpha != nullptr) ? getValue<float>(alpha) : 1.0;

    if (value == 1.0) {
        AclOpRunner<2, 1, dtypeConvertor>("AddV2").addInput(input, other).addOutput(out).run(ctx);
    } else {
        AclOpRunner<2, 1>("AxpyV2")
            .addInput(input, ACL_FORMAT_ND)
            .addInput(other, ACL_FORMAT_ND)
            .setAttr<float>("alpha", value)
            .addOutput(out, ACL_FORMAT_ND)
            .run(ctx);
    }
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiAdd(ctx, input, input, other, alpha);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t
diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    const float value = (alpha != nullptr) ? getValue<float>(alpha) : 1.0;
    AclOpRunner<1, 1>("Adds").addInput(input).setAttr<float>("value", value * getValue<float>(other)).addOutput(out).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t input,
                                                    const diopiScalar_t* other,
                                                    const diopiScalar_t* alpha) {
    diopiAddScalar(ctx, input, input, other, alpha);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
