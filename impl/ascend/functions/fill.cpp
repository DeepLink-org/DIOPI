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

    if (shape.len == 0) {
        int64_t sizeTmp[1] = {1};
        shape = arrayToDiopiSize(sizeTmp, 1);
        int64_t elemsize;
        diopiGetTensorElemSize(input, &elemsize);
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        void *src, *dst;
        diopiScalar_t scalar;
        scalar.stype = diopi_dtype_float64;
        scalar.fval = val;
        if (diopi_dtype_float16 == dtype || diopi_dtype_int16 == dtype) {
            diopiTensorHandle_t inputTemp;
            makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
            makeTensorFromScalar(ctx, &scalar, &inputCopy, diopi_dtype_float32, diopi_host);
            diopiGetTensorData(inputTemp, &dst);
            diopiGetTensorData(inputCopy, &src);
            diopiGetTensorElemSize(inputTemp, &elemsize);
            CALL_ACLRT(aclrtMemcpyAsync(dst, elemsize, src, elemsize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
            CALL_ACLRT(aclrtSynchronizeStream(stream));
            diopiCastDtype(ctx, input, inputTemp);
        } else {
            makeTensorFromScalar(ctx, &scalar, &inputCopy, dtype, diopi_host);
            diopiGetTensorData(input, &dst);
            diopiGetTensorData(inputCopy, &src);
            CALL_ACLRT(aclrtMemcpyAsync(dst, elemsize, src, elemsize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
            CALL_ACLRT(aclrtSynchronizeStream(stream));
        }
    } else {
        if (diopi_dtype_bool == dtype) {
            makeTensorLike(ctx, &inputCopy, input, diopi_dtype_int32);
            AclOpRunner<1, 1>("Fills", ctx).addInput(inputCopy).setAttr<float>("value", val).addOutput(inputCopy).run();
            diopiCastDtype(ctx, input, inputCopy);
        } else {
            AclOpRunner<1, 1>("Fills", ctx).addInput(input).setAttr<float>("value", val).addOutput(input).run();
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
