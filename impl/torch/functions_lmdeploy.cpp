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

    std::cout << "LXZ: max_input_length " << max_input_length << std::endl;
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

/**
 * @brief EmbeddingLookupPosEncoding. Find id in embedding_table and get [hidden], only this step
 * @param[in] ctx The diopi context.
 * @param[out] from_tensor : Output ids.shape = [batch_size, hidden].type = [float32, float16]
 * @param[in] embedding_table : Embedding table.shape=[vocab, hidden].type = [float32, float16]
 * @param[in] all_ids : Input ids.shape=[batch_size, sessionlen].type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] hidden_units : Hidden units.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t embedding_table,
                                                       diopiConstTensorHandle_t all_ids, const int64_t batch_size, const int64_t hidden_units,
                                                       const int64_t step) {
    if (from_tensor == nullptr || embedding_table == nullptr || all_ids == nullptr) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;

    diopiGetTensorDtype(all_ids, &in_type);
    diopiGetTensorShape(all_ids, &in_shape);
    diopiGetTensorStride(all_ids, &in_stride);
    diopiGetTensorDevice(all_ids, &in_device);

    diopiTensorHandle_t this_step_ids;
    diopiSize_t this_step_shape;
    this_step_shape.len = 1;
    this_step_shape.data = &batch_size;
    diopiSize_t this_step_stride;
    this_step_stride.len = 1;
    this_step_stride.data = in_stride.data + 1;
    diopiRequireTensor(ctx, &this_step_ids, &this_step_shape, &this_step_stride, in_type, in_device);

    diopiSelect(ctx, this_step_ids, all_ids, 0, step);
    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, this_step_ids);
    return diopiSuccess;
}

/**
 * @brief InputIdsEmbeddingLookupPosEncoding. Find id in embedding_table and get [hidden].
 * @param[in] ctx The diopi context.
 * @param[out] from_tensor : Output ids.shape = [input_lengths, hidden].type = [float32, float16]
 * @param[in] input_ids : Input ids.shape=[input_lengths].type = [int64, int32]
 * @param[in] embedding_table : Embedding table.shape=[vocab, hidden].type = [float32, float16]
 * @param[in] input_lengths : Input lengths.type = [int64, int32]
 * @param[in] hidden_units : Hidden units.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiInputIdsEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t input_ids,
                                                               diopiConstTensorHandle_t embedding_table, const int64_t input_lengths,
                                                               const int64_t hidden_units) {
    if (from_tensor == nullptr || input_ids == nullptr || embedding_table == nullptr) {
        return diopiErrorOccurred;
    }

    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, input_ids);
    return diopiSuccess;
}
/**
 * @brief StopWordsCriterion. Judging the end situation based on stopword.
 * get base_stop_words and base_offsets from stop_words for each batch.
 * every item in base_offsets means item-end, and they also the item-start of the next item, for the first item item-start is 0.
 * If time-size = end - start < step+1, then check item.
 * for (int token_idx = item_size - 1; token_idx >= 0; token_idx--) {const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size +
 * id_offset + batch_idx];if (previous_token != base_stop_words[item_start + token_idx]) {should_stop = false; break;}} if one batch is should_stop, then it is
 * finished.
 * @param[in] ctx The diopi context.
 * @param[in] output_ids : Output ids.shape = [step, batch_size].type = [int64, int32]
 * @param[in] stop_words : Stop words list.shape = [batch_size, 2, stop_words_len].type = [int64, int32]
 * @param[inout] finished : Finished.shape = [batch_size].type = [bool]
 * @param[in] id_offset : Offset of output_ids.type = [int64, int32]
 * @param[in] stop_words_len : Stop words len tensor.type = [int64, int32]
 * @param[in] batch_size : batch_size.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiStopWordsCriterion(diopiContextHandle_t ctx, diopiConstTensorHandle_t output_ids, diopiConstTensorHandle_t stop_words,
                                               diopiTensorHandle_t finished, int64_t id_offset, int64_t stop_words_len, int64_t batch_size, int64_t step) {
    // always id_offset = 0
    if (output_ids == nullptr || stop_words == nullptr || finished == nullptr) {
        return diopiErrorOccurred;
    }

    diopiTensorHandle_t stop_words_host;
    diopiSize_t stop_words_shape, stop_words_stride;
    diopiGetTensorShape(stop_words, &stop_words_shape);
    diopiGetTensorStride(stop_words, &stop_words_stride);
    diopiRequireTensor(ctx, &stop_words_host, &stop_words_shape, &stop_words_stride, diopi_dtype_int32, diopi_host);
    diopiCopyD2H(ctx, stop_words_host, stop_words, false);

    const int32_t *stop_words_ptr;
    int32_t* stop_words_host_ptr;

    diopiGetTensorDataConst(stop_words, reinterpret_cast<const void **>(&stop_words_ptr));
    diopiGetTensorData(stop_words_host, reinterpret_cast<void **>(&stop_words_host_ptr));

    for(int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t* base_stop_words_host = stop_words_host_ptr + batch_idx * 2 * stop_words_len;
        const int32_t* base_offsets_host    = base_stop_words_host + stop_words_len;
        const int32_t* base_stop_word = stop_words_ptr + batch_idx * 2 * stop_words_len;
        const int32_t* base_offsets   = base_stop_word + stop_words_len;        

        for (int64_t stop_word_idx = 0; stop_word_idx < stop_words_len; ++stop_word_idx) {
            if (base_stop_words_host[stop_word_idx] < 0) {
                continue;
            }
            const int32_t stop_word_start_idx = (stop_word_idx > 0) ? base_offsets_host[stop_word_idx - 1] : 0;
            const int32_t stop_word_end_idx   = base_offsets_host[stop_word_idx] - 1;
            const int64_t stop_word_len  = stop_word_end_idx - stop_word_start_idx + 1;

            if (step + 1 < stop_word_len) {
                continue;
            }

            diopiTensorHandle_t stop_word_tensor;
            diopiSize_t stop_word_shape;
            stop_word_shape.len = 1;
            stop_word_shape.data = &stop_word_len;
            diopiDevice_t stop_word_device;
            diopiGetTensorDevice(stop_words, &stop_word_device);
            diopiSize_t stride;
            stride.len = -1;
            stride.data = reinterpret_cast<const int64_t *>(stop_words_ptr + stop_word_start_idx);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  3.3"<< std::endl;
            diopiRequireTensor(ctx, &stop_word_tensor, &stop_word_shape, &stride, diopi_dtype_int32, stop_word_device);

            // 转置一下
            // diopiTensorHandle_t output_ids_transpose;
            // diopiSize_t output_ids_transpose_shape;
            // output_ids_transpose_shape.len = 2;
            // output_ids_transpose_shape.data = new int64_t[2]{batch_size, step};
            // diopiSize_t output_ids_transpose_stride;
            // output_ids_transpose_stride.len = 2;
            // output_ids_transpose_stride.data = new int64_t[2]{step * static_cast<int64_t>(sizeof(int)), sizeof(int)};
            // diopiRequireTensor(ctx, &output_ids_transpose, &output_ids_transpose_shape, &output_ids_transpose_stride, diopi_dtype_int32, stop_word_device);
            // diopiTranspose(ctx, output_ids_transpose, output_ids, 0, 1);

            // 拿到那一列
            diopiTensorHandle_t output_ids_col;
            diopiGetTensorStride(stop_word_tensor, &stride);
            diopiSize_t output_ids_col_shape;
            output_ids_col_shape.len = 1;
            output_ids_col_shape.data = &step;
            diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, diopi_dtype_int32, diopi_device);
            diopiSelect(ctx, output_ids_col, output_ids, 1, batch_idx);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  3.4"<< std::endl;

            

            // 拿到那一列中需要比较的部分
            char *output_ids_col_data;
            diopiGetTensorData(output_ids_col, reinterpret_cast<void **>(&output_ids_col_data));
            int64_t elesize;
            diopiGetTensorElemSize(output_ids_col, &elesize);
            std::cout << "step: " << step << "index: " << step - stop_word_len + 1 << " elesize: " << elesize << std::endl;

            output_ids_col_data += (step - stop_word_len) * elesize;
            diopiSize_t output_ids_to_compare_shape;
            output_ids_to_compare_shape.len = 1;
            output_ids_to_compare_shape.data = &stop_word_len;
            // diopiGetTensorStride(stop_word_tensor, &stride);
            stride.len = -1;
            stride.data = reinterpret_cast<const int64_t *>(reinterpret_cast<int64_t *>(output_ids_col_data));
            diopiTensorHandle_t output_ids_to_compare;
            std::cout << "LXZ diopiStopWordsCriterion LOG:  4"<< std::endl;
            diopiRequireTensor(ctx, &output_ids_to_compare, &output_ids_to_compare_shape, &stride, diopi_dtype_int32, diopi_device);
            diopiGetTensorData(output_ids_to_compare, reinterpret_cast<void **>(&output_ids_col_data));


            // for test
            // diopiTensorHandle_t output_ids_to_compare_host;
            // diopiGetTensorStride(stop_word_tensor, &stride);
            // diopiRequireTensor(ctx, &output_ids_to_compare_host, &stop_word_shape, &stride, diopi_dtype_int32, diopi_host);
            // std::cout << "LXZ diopiStopWordsCriterion LOG:  4.1"<< std::endl;
            // diopiCopyD2H(ctx, output_ids_to_compare_host, output_ids_to_compare, false);
            // int *output_ids_to_compare_host_data;
            // diopiGetTensorData(output_ids_to_compare_host, reinterpret_cast<void **>(&output_ids_to_compare_host_data));
            // std::cout << "LXZ diopiStopWordsCriterion LOG:  4.2 " << output_ids_to_compare_host_data[0] << std::endl;

            // diopiTensorHandle_t stop_word_tensor_host;
            // diopiGetTensorStride(stop_word_tensor, &stride);
            // diopiRequireTensor(ctx, &stop_word_tensor_host, &stop_word_shape, &stride, diopi_dtype_int32, diopi_host);
            // std::cout << "LXZ diopiStopWordsCriterion LOG:  4.1"<< std::endl;
            // diopiCopyD2H(ctx, stop_word_tensor_host, stop_word_tensor, false);
            // int *stop_word_tensor_host_data;
            // diopiGetTensorData(stop_word_tensor_host, reinterpret_cast<void **>(&stop_word_tensor_host_data));
            // std::cout << "LXZ diopiStopWordsCriterion LOG:  4.2 " << stop_word_tensor_host_data[0] << std::endl;
            // test end


            // std::cout << "start " << step - stop_word_len + 1 << " end " << step << std::endl;
            // diopiSlice(ctx, output_ids_to_compare, output_ids_col, 0, step - stop_word_len + 1, step, 1);
            // diopiSlice(ctx, output_ids_to_compare, output_ids_col, 0, 0, 1, 1);
            
            diopiTensorHandle_t cmp_res;

            diopiSize_t finished_stride;
            diopiGetTensorStride(finished, &finished_stride);
            
            diopiRequireTensor(ctx, &cmp_res, &stop_word_shape, nullptr, diopi_dtype_bool, stop_word_device);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  5"<< std::endl;
            diopiEq(ctx, cmp_res, output_ids_to_compare, stop_word_tensor);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  6"<< std::endl;
            diopiTensorHandle_t cmp_res_sum;
            diopiSize_t cmp_res_sum_shape;
            cmp_res_sum_shape.len = 1;
            int64_t tmp_one = 1;
            cmp_res_sum_shape.data = &tmp_one;

            diopiRequireTensor(ctx, &cmp_res_sum, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, stop_word_device);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  6.1"<< std::endl;
            int64_t tmp_zero = 0;
            diopiAll(ctx, cmp_res_sum, cmp_res, &tmp_zero);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  7"<< std::endl;
            diopiTensorHandle_t cmp_res_sum_host;
            diopiRequireTensor(ctx, &cmp_res_sum_host, &cmp_res_sum_shape, &finished_stride, diopi_dtype_bool, diopi_host);
            diopiCopyD2H(ctx, cmp_res_sum_host, cmp_res_sum, false);
            std::cout << "LXZ diopiStopWordsCriterion LOG:  8"<< std::endl;
            bool* cmp_res_sum_host_data;
            diopiGetTensorData(cmp_res_sum_host, reinterpret_cast<void **>(&cmp_res_sum_host_data));
            if (cmp_res_sum_host_data[0]) {
                diopiTensorHandle_t batch_idx_tensor;
                diopiSize_t batch_idx_shape, batch_idx_stride;
                batch_idx_shape.len = 1;
                batch_idx_shape.data = &tmp_one;
                int64_t batch_idx_stride_tmp = sizeof(int);
                batch_idx_stride.len = 1;
                batch_idx_stride.data = &batch_idx_stride_tmp;
                diopiRequireTensor(ctx, &batch_idx_tensor, &batch_idx_shape, &batch_idx_stride, diopi_dtype_int32, stop_word_device);

                diopiScalar_t batch_idx_scalar;
                batch_idx_scalar.stype = diopi_dtype_int64;
                batch_idx_scalar.ival = batch_idx;
                diopiFill(ctx, batch_idx_tensor, &batch_idx_scalar);

                diopiIndexFillInp(ctx, finished, 0, batch_idx_tensor, cmp_res_sum);
                break;
            }
            std::cout << "LXZ diopiStopWordsCriterion LOG:  9"<< std::endl;
        }   
    }
    return diopiSuccess;
}

}  // extern "C"
