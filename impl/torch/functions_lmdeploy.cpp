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
    front_stride.data = reinterpret_cast<const int64_t*>(input_data);
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


/**
 * @brief Total_padding_count.Padding_count = maxinputlen - inputlen for each batch.
 * @param[in] ctx The diopi context.
 * @param[out] total_padding_count : Total padding_count.shape=[batch_size].type = [int64, int32]
 * @param[in] input_lengths : Input length.shape=[batch_size].type = [int64, int32]
 * @param[in] max_input_length : Max input length.type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiUpdatePaddingCount(diopiContextHandle_t ctx, diopiTensorHandle_t total_padding_count, diopiConstTensorHandle_t input_lengths,
                                               int64_t max_input_length, int64_t batch_size) {
    
    if (input_lengths == nullptr) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;
    diopiGetTensorDtype(input_lengths, &in_type);
    diopiGetTensorShape(input_lengths, &in_shape);
    diopiGetTensorStride(input_lengths, &in_stride);
    diopiGetTensorDevice(input_lengths, &in_device);

    diopiScalar_t max_input_length_scalar;
    max_input_length_scalar.stype = diopi_dtype_int64;
    max_input_length_scalar.ival = max_input_length;
    
    std::cout << "LXZ: max_input_length "<< max_input_length << std::endl;
    if (total_padding_count == nullptr) {
        std::cout << "LXZ: nullptr " << std::endl;
        diopiRequireTensor(ctx, &total_padding_count, &in_shape, &in_stride, in_type, in_device);
    }
    
    diopiFill(ctx, total_padding_count, &max_input_length_scalar);
    
    std::cout << "LXZ: start diopiSubInp " << std::endl;

    diopiScalar_t one;
    one.stype = diopi_dtype_int64;
    one.ival = 1;
    diopiSubInp(ctx, total_padding_count, const_cast<diopiConstTensorHandle_t>(input_lengths), &one);
    std::cout << "LXZ: end diopiSubInp" << std::endl;
    return diopiSuccess;
}

/**
 * @brief LengthCriterion. Judging and counting the end situation based on length.If all fin then should_stop.
 * @param[in] ctx The diopi context.
 * @param[inout] finished : Finished.shape = [batch_size].type = [bool]
 * @param[out] should_stop : If all fin then should_stop.shape = [1].type = [bool]
 * @param[out] finished_sum : Total finished.shape = [1].type = [int64, int32]
 * @param[in] sequence_limit_length : Sequence limit length tensor.shape = [batch_size].type = [int64, int32]
 * @param[in] batch_size : Input tensor.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiLengthCriterion(diopiContextHandle_t ctx, diopiTensorHandle_t finished, diopiTensorHandle_t should_stop,
                                            diopiTensorHandle_t finished_sum, diopiConstTensorHandle_t sequence_limit_length, int64_t batch_size, int64_t step) {
    if (finished == nullptr || sequence_limit_length == nullptr) {
        return diopiErrorOccurred;                                                  
    }
    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;
    diopiGetTensorDtype(finished, &in_type);
    diopiGetTensorShape(finished, &in_shape);
    diopiGetTensorStride(finished, &in_stride);
    diopiGetTensorDevice(finished, &in_device);

    diopiScalar_t batch_size_scalar;
    batch_size_scalar.stype = diopi_dtype_int64;
    batch_size_scalar.ival = batch_size;
    
    // diopiTensorHandle_t tmp;
    // diopiRequireTensor(ctx, &tmp, &in_shape, &in_stride, in_type, in_device);

    diopiScalar_t step_scalar;
    step_scalar.stype = diopi_dtype_int64;
    step_scalar.ival = step + 1;
    
    // if step >= sequence_limit_length, then finished = true
    // diopiSubInpScalar(ctx, sequence_limit_length, &step_scalar);

    diopiLeScalar(ctx, finished, sequence_limit_length, &step_scalar);
    diopiSize_t dim_zero;
    int64_t tmp_zero = 0;
    dim_zero.data = &tmp_zero;
    dim_zero.len = 1;
    
    diopiSum(ctx, finished_sum, finished, dim_zero);
    diopiEqScalar(ctx, should_stop, finished_sum, &batch_size_scalar);
    
    return diopiSuccess;
} 

} // extern "C"
