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

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *value) {
    int64_t numel = 0;
    diopiSize_t shape;
    diopiGetTensorNumel(input, &numel);
    diopiGetTensorShape(input, &shape);
    if (numel <= 0) {
        return diopiSuccess;
    }
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
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiTensorHandle_t inputCopy;
    if (shape.len == 0) {
        int64_t sizeTmp[1] = {1};
        shape = arrayToDiopiSize(sizeTmp, 1);
        int64_t elemsize;
        diopiStreamHandle_t stream;
        diopiGetTensorElemSize(input, &elemsize);
        diopiGetStream(ctx, &stream);
        void *src, *dst;
        diopiScalar_t scalar;
        scalar.stype = diopi_dtype_float64;
        scalar.fval = val;
        makeTensorFromScalar(ctx, &scalar, &inputCopy, dtype, diopi_host);
        diopiGetTensorData(input, &dst);
        diopiGetTensorData(inputCopy, &src);
        CALL_ACLRT(aclrtMemcpyAsync(dst, elemsize, src, elemsize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        CALL_ACLRT(aclrtSynchronizeStream(stream));
    } else {
        if (dtype == diopi_dtype_int8 || dtype == diopi_dtype_uint8) {
            makeTensorLike(ctx, &inputCopy, input, diopi_dtype_int32);
        } else {
            inputCopy = input;
        }
        AclOpRunner<1, 1>("Fills", ctx).addInput(inputCopy).setAttr<float>("value", val).addOutput(inputCopy).run();
        if (dtype == diopi_dtype_int8 || dtype == diopi_dtype_uint8) {
            diopiCastDtype(ctx, input, inputCopy);
        }
    }
    auto zeroValueScalar = diopiScalar_t();
    zeroValueScalar.stype = diopi_dtype_float64;
    zeroValueScalar.fval = 0.0;
    if (divByZero) diopiDivInpScalar(ctx, input, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
