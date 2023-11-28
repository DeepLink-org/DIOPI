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

DIOPI_API diopiError_t diopiLengthCriterion(diopiContextHandle_t ctx, diopiTensorHandle_t finished, diopiTensorHandle_t should_stop,
                                            diopiTensorHandle_t finished_sum, diopiConstTensorHandle_t sequence_limit_length, int64_t batch_size,
                                            int64_t step) {
    if (finished == nullptr || sequence_limit_length == nullptr) {
        return diopiErrorOccurred;
    }

    diopiScalar_t step_scalar;
    step_scalar.stype = diopi_dtype_int64;
    step_scalar.ival = step;

    diopiLeScalar(ctx, finished, sequence_limit_length, &step_scalar);

    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;
    diopiGetTensorDtype(finished, &in_type);
    diopiGetTensorShape(finished, &in_shape);
    diopiGetTensorStride(finished, &in_stride);
    diopiGetTensorDevice(finished, &in_device);
    diopiTensorHandle_t finished_fp64;
    diopiRequireTensor(ctx, &finished_fp64, &in_shape, &in_stride, diopi_dtype_float64, in_device);
    diopiCastDtype(ctx, finished_fp64, finished);

    diopiGetTensorShape(finished_sum, &in_shape);
    diopiGetTensorStride(finished_sum, &in_stride);
    diopiGetTensorDevice(finished_sum, &in_device);
    diopiTensorHandle_t finished_sum_device;
    diopiTensorHandle_t finished_sum_fp64_device;
    diopiRequireTensor(ctx, &finished_sum_device, &in_shape, &in_stride, in_type, diopi_device);
    diopiRequireTensor(ctx, &finished_sum_fp64_device, &in_shape, &in_stride, diopi_dtype_float64, diopi_device);
    diopiCopyH2D(ctx, finished_sum_device, finished_sum, false);
    diopiCastDtype(ctx, finished_sum_fp64_device, finished_sum_device);

    diopiSize_t dim_zero;
    int64_t tmp_zero = 0;
    dim_zero.data = &tmp_zero;
    dim_zero.len = 1;
    diopiSum(ctx, finished_sum_fp64_device, finished_fp64, dim_zero);

    diopiCastDtype(ctx, finished_sum_device, finished_sum_fp64_device);
    diopiCopyD2H(ctx, finished_sum, finished_sum_device, false);

    diopiGetTensorDtype(finished, &in_type);
    diopiGetTensorShape(finished, &in_shape);
    diopiGetTensorStride(finished, &in_stride);
    diopiGetTensorDevice(finished, &in_device);
    diopiTensorHandle_t h_finished;
    diopiRequireTensor(ctx, &h_finished, &in_shape, &in_stride, in_type, diopi_host);
    diopiCopyD2H(ctx, h_finished, finished, false);
    diopiAll(ctx, should_stop, h_finished, &tmp_zero);
    return diopiSuccess;
}

}  // extern "C"
