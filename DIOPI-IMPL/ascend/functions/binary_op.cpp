#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t
diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {

    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiAdd(ctx, input, input, other, alpha);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t
diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {

    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t input,
                                                    const diopiScalar_t* other,
                                                    const diopiScalar_t* alpha) {

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
