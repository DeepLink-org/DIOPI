/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>

#include <iostream>
#include <vector>

extern "C" {

DIOPI_API diopiError_t diopiPlusScalarInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, const int64_t val, const int64_t size) {
    diopiSize_t in_shape;
    diopiGetTensorShape(inoutput, &in_shape);
    if (in_shape.len != 1) {
        return diopiErrorOccurred;
    }
    int64_t input_len = in_shape.data[0];

    diopiDtype_t in_type;
    diopiSize_t in_stride;
    diopiDevice_t in_device;
    diopiGetTensorDtype(inoutput, &in_type);
    diopiGetTensorStride(inoutput, &in_stride);
    diopiGetTensorDevice(inoutput, &in_device);

    diopiSize_t front_shape;
    front_shape.data = &size;
    front_shape.len = 1;
    diopiTensorHandle_t tmp[2];
    diopiRequireTensor(ctx, tmp, &front_shape, &in_stride, in_type, in_device);

    diopiScalar_t front_scalar;
    front_scalar.stype = diopi_dtype_int64;
    front_scalar.ival = val;
    diopiFill(ctx, tmp[0], &front_scalar);

    diopiTensorHandle_t added = tmp[0];

    if (size < input_len) {
        int64_t latter_len = input_len - size;
        diopiSize_t latter_shape;
        latter_shape.data = &latter_len;
        latter_shape.len = 1;
        diopiRequireTensor(ctx, &tmp[1], &latter_shape, &in_stride, in_type, in_device);

        diopiScalar_t latter_scalar;
        latter_scalar.stype = diopi_dtype_int64;
        latter_scalar.ival = 0;
        diopiFill(ctx, tmp[1], &latter_scalar);

        diopiRequireTensor(ctx, &added, &in_shape, &in_stride, in_type, in_device);
        diopiCat(ctx, added, const_cast<diopiConstTensorHandle_t*>(&tmp[0]), 2, 0);
    }

    diopiScalar_t one;
    one.stype = diopi_dtype_int64;
    one.ival = 1;
    diopiAddInp(ctx, inoutput, added, &one);
    return diopiSuccess;
}

}  // extern "C"