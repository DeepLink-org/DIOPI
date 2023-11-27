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

/**
 * @brief PlusScalar. if 0 < index < size, add val.
 * @param[in] ctx The diopi context.
 * @param[inout] inoutput : Output tensor.shape=[len].type = [int64, int32]
 * @param[in] val : Val for add.type = [int64, int32]
 * @param[in] size : Size or maxindex.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiPlusScalarInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, const int64_t val, const int64_t size) {
    diopiSize_t in_shape;
    diopiGetTensorShape(inoutput, &in_shape);
    if (in_shape.len != 1) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiDevice_t in_device;
    diopiGetTensorDtype(inoutput, &in_type);
    diopiGetTensorDevice(inoutput, &in_device);

    int64_t front_len = (size <= in_shape.data[0]) ? size : in_shape.data[0];
    diopiSize_t front_shape;
    front_shape.data = &front_len;
    front_shape.len = 1;
    diopiSize_t front_stride;

    void *input_data;
    diopiGetTensorData(inoutput, &input_data);
    front_stride.data = reinterpret_cast<const int64_t *>(input_data);
    front_stride.len = -1;
    diopiTensorHandle_t front;
    diopiRequireTensor(ctx, &front, &front_shape, &front_stride, in_type, in_device);

    diopiScalar_t val_scalar;
    val_scalar.stype = diopi_dtype_int64;
    val_scalar.ival = val;

    diopiScalar_t one;
    one.stype = diopi_dtype_int64;
    one.ival = 1;
    diopiAddInpScalar(ctx, front, &val_scalar, &one);
    return diopiSuccess;
}

}  // extern "C"