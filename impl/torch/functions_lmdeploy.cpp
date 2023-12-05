/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>

#include <iostream>
#include <vector>
#include <cmath>

#include "context.h"
#include "helper.hpp"
#include "vision_kernel.h"

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

    diopiDtype_t ids_type;
    diopiGetTensorDtype(output_ids, &ids_type);
    if (ids_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }
    
    for(int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t* base_stop_words_host = stop_words_host_ptr + batch_idx * 2 * stop_words_len;
        const int32_t* base_offsets_host    = base_stop_words_host + stop_words_len;
        const int32_t* base_stop_word = stop_words_ptr + batch_idx * 2 * stop_words_len;       

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
            stride.data = reinterpret_cast<const int64_t *>(base_stop_word + stop_word_start_idx);
            diopiRequireTensor(ctx, &stop_word_tensor, &stop_word_shape, &stride, ids_type, stop_word_device);

            diopiTensorHandle_t output_ids_col;
            diopiGetTensorStride(stop_word_tensor, &stride);
            diopiSize_t output_ids_col_shape;
            output_ids_col_shape.len = 1;
            output_ids_col_shape.data = &step;
            diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, ids_type, diopi_device);
            diopiSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

            char *output_ids_col_data;
            diopiGetTensorData(output_ids_col, reinterpret_cast<void **>(&output_ids_col_data));
            int64_t elem_size;
            diopiGetTensorElemSize(output_ids_col, &elem_size);
            output_ids_col_data += (step - stop_word_len) * elem_size;
            stride.len = -1;
            stride.data = reinterpret_cast<const int64_t *>(reinterpret_cast<int64_t *>(output_ids_col_data));
            diopiTensorHandle_t output_ids_to_compare;
            diopiRequireTensor(ctx, &output_ids_to_compare, &stop_word_shape, &stride, ids_type, diopi_device);

            diopiTensorHandle_t cmp_res;
            diopiRequireTensor(ctx, &cmp_res, &stop_word_shape, nullptr, diopi_dtype_bool, stop_word_device);
            diopiEq(ctx, cmp_res, output_ids_to_compare, stop_word_tensor);
            
            diopiTensorHandle_t cmp_res_sum;
            diopiSize_t cmp_res_sum_shape;
            cmp_res_sum_shape.len = 1;
            int64_t tmp_one = 1;
            cmp_res_sum_shape.data = &tmp_one;
            diopiRequireTensor(ctx, &cmp_res_sum, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, stop_word_device);
            int64_t tmp_zero = 0;
            diopiAll(ctx, cmp_res_sum, cmp_res, &tmp_zero);
            
            diopiTensorHandle_t cmp_res_sum_host;
            diopiRequireTensor(ctx, &cmp_res_sum_host, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, diopi_host);
            diopiCopyD2H(ctx, cmp_res_sum_host, cmp_res_sum, false);
            bool* cmp_res_sum_host_data;
            diopiGetTensorData(cmp_res_sum_host, reinterpret_cast<void **>(&cmp_res_sum_host_data));
            
            if (cmp_res_sum_host_data[0]) {
                diopiTensorHandle_t batch_idx_tensor;
                diopiSize_t batch_idx_shape;
                batch_idx_shape.len = 1;
                batch_idx_shape.data = &tmp_one;
                diopiRequireTensor(ctx, &batch_idx_tensor, &batch_idx_shape, nullptr, diopi_dtype_int32, stop_word_device);

                diopiScalar_t batch_idx_scalar;
                batch_idx_scalar.stype = diopi_dtype_int64;
                batch_idx_scalar.ival = batch_idx;
                diopiFill(ctx, batch_idx_tensor, &batch_idx_scalar);
                diopiIndexFillInp(ctx, finished, 0, batch_idx_tensor, cmp_res_sum);
                break;
            }
        }   
    }
    return diopiSuccess;
}


/**
 * @brief BanBadWords.
 * get base_bad_words and base_offsets from stop_words for each batch.
 * every item in base_offsets means item-end, and they also the item-start of the next item, for the first item item-start is 0.
 * If time-size = end - start < step+1, then check item.
 * for (int token_idx = item_size - 1; token_idx >= 0; token_idx--) {const int previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size +
 * id_offset + batch_idx];if (previous_token != base_stop_words[item_start + token_idx]) {should_ban = false; break;}} if this tiem should_ban, then get banid =
 * base_bad_words[item-end - 1]. if 0 < banid < vocab_size then logits ban id in this batch is set to -INFINITY
 * @param[in] ctx The diopi context.
 * @param[inout] logits : Output logits.shape = [batch_size, vocab_size].type = [float32, float16]
 * @param[in] output_ids : Output ids.shape = [step, batch].type = [int64, int32]
 * @param[in] bad_words : Stop words list.shape = [batch_size, 2, stop_words_len] or [2, stop_words_len] for share.type = [int64, int32]
 * @param[in] id_offset : Offset of output_ids.type = [int64, int32]
 * @param[in] bad_words_len : Stop words len.type = [int64, int32]
 * @param[in] share_words : Stop words is shared or not.type = [bool]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 * @param[in] vocab_size : Vocab size.type = [int64, int32]
 * @param[in] step : Step.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiBanBadWordsInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t output_ids,
                                           diopiConstTensorHandle_t bad_words, int64_t id_offset, int64_t bad_words_len, bool share_words, int64_t batch_size,
                                           int64_t vocab_size, int64_t step) {
    // always id_offset = 0
    if (logits == nullptr || output_ids == nullptr || bad_words == nullptr) {
        return diopiErrorOccurred;
    }

    diopiTensorHandle_t bad_words_host;
    diopiSize_t bad_words_shape, bad_words_stride;
    diopiGetTensorShape(bad_words, &bad_words_shape);
    diopiGetTensorStride(bad_words, &bad_words_stride);
    diopiRequireTensor(ctx, &bad_words_host, &bad_words_shape, &bad_words_stride, diopi_dtype_int32, diopi_host);
    diopiCopyD2H(ctx, bad_words_host, bad_words, false);

    const int32_t *bad_words_ptr;
    int32_t* bad_words_host_ptr;

    diopiGetTensorDataConst(bad_words, reinterpret_cast<const void **>(&bad_words_ptr));
    diopiGetTensorData(bad_words_host, reinterpret_cast<void **>(&bad_words_host_ptr));

    diopiDtype_t ids_type;
    diopiGetTensorDtype(output_ids, &ids_type);
    if (ids_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }
    
    for(int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t* base_bad_words_host = share_words ? bad_words_host_ptr : bad_words_host_ptr + batch_idx * 2 * bad_words_len;
        const int32_t* base_offsets_host    = base_bad_words_host + bad_words_len;
        const int32_t* base_bad_word = share_words ? bad_words_ptr : bad_words_ptr + batch_idx * 2 * bad_words_len;       

        for (int64_t bad_word_idx = 0; bad_word_idx < bad_words_len; ++bad_word_idx) {
            if (base_bad_words_host[bad_word_idx] < 0) {
                continue;
            }
            const int32_t bad_word_start_idx = (bad_word_idx > 0) ? base_offsets_host[bad_word_idx - 1] : 0;
            const int32_t bad_word_end_idx   = base_offsets_host[bad_word_idx] - 1;
            const int64_t bad_word_len  = bad_word_end_idx - bad_word_start_idx + 1;

            

            if (step + 1 < bad_word_len) {
                continue;
            }

            bool* cmp_res_sum_host_data = nullptr;
            if (bad_word_len != 1) {
                diopiTensorHandle_t bad_word_tensor;
                diopiSize_t bad_word_shape;
                bad_word_shape.len = 1;
                bad_word_shape.data = &bad_word_len;
                diopiDevice_t bad_word_device;
                diopiGetTensorDevice(bad_words, &bad_word_device);
                diopiSize_t stride;
                stride.len = -1;
                stride.data = reinterpret_cast<const int64_t *>(base_bad_word + bad_word_start_idx);
                diopiRequireTensor(ctx, &bad_word_tensor, &bad_word_shape, &stride, ids_type, bad_word_device);

                diopiTensorHandle_t output_ids_col;
                diopiGetTensorStride(bad_word_tensor, &stride);
                diopiSize_t output_ids_col_shape;
                output_ids_col_shape.len = 1;
                output_ids_col_shape.data = &step;
                diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, ids_type, diopi_device);
                diopiSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

                char *output_ids_col_data;
                diopiGetTensorData(output_ids_col, reinterpret_cast<void **>(&output_ids_col_data));
                int64_t elem_size;
                diopiGetTensorElemSize(output_ids_col, &elem_size);
                output_ids_col_data += (step - bad_word_len) * elem_size;

                stride.len = -1;
                stride.data = reinterpret_cast<const int64_t *>(reinterpret_cast<int64_t *>(output_ids_col_data));
                diopiTensorHandle_t output_ids_to_compare;
                diopiRequireTensor(ctx, &output_ids_to_compare, &bad_word_shape, &stride, ids_type, diopi_device);

                diopiTensorHandle_t cmp_res;
                diopiRequireTensor(ctx, &cmp_res, &bad_word_shape, nullptr, diopi_dtype_bool, bad_word_device);
                diopiEq(ctx, cmp_res, output_ids_to_compare, bad_word_tensor);
                
                diopiTensorHandle_t cmp_res_sum;
                diopiSize_t cmp_res_sum_shape;
                cmp_res_sum_shape.len = 1;
                int64_t tmp_one = 1;
                cmp_res_sum_shape.data = &tmp_one;
                diopiRequireTensor(ctx, &cmp_res_sum, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, bad_word_device);
                int64_t tmp_zero = 0;
                diopiAll(ctx, cmp_res_sum, cmp_res, &tmp_zero);
                
                diopiTensorHandle_t cmp_res_sum_host;
                diopiRequireTensor(ctx, &cmp_res_sum_host, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, diopi_host);
                diopiCopyD2H(ctx, cmp_res_sum_host, cmp_res_sum, false);
                
                diopiGetTensorData(cmp_res_sum_host, reinterpret_cast<void **>(&cmp_res_sum_host_data));
            }

            if (bad_word_len == 1 || (cmp_res_sum_host_data != nullptr && cmp_res_sum_host_data[0])) {
                int32_t banned_token = base_bad_words_host[bad_word_end_idx];
                if (0 < banned_token && banned_token < vocab_size) {
                    diopiTensorHandle_t banned_token_tensor;
                    diopiSize_t banned_token_shape;
                    banned_token_shape.len = 1;
                    int64_t tmp_one = 1;
                    banned_token_shape.data = &tmp_one;
                    diopiSize_t stride;
                    stride.len = -1;
                    stride.data = reinterpret_cast<const int64_t *>(base_bad_word + bad_word_end_idx);
                    diopiRequireTensor(ctx, &banned_token_tensor, &banned_token_shape, &stride, ids_type, diopi_device);
                    
                    diopiTensorHandle_t logit_to_modify;
                    diopiDtype_t logits_type;
                    diopiGetTensorDtype(logits, &logits_type);
                    diopiSize_t logit_to_modify_shape, logit_to_modify_stride;
                    logit_to_modify_shape.len = 1;
                    logit_to_modify_shape.data = &tmp_one;
                    logit_to_modify_stride.len = -1;
                    char* logit_to_modify_data;
                    diopiGetTensorData(logits, reinterpret_cast<void **>(&logit_to_modify_data));
                    int64_t elem_size;
                    diopiGetTensorElemSize(logits, &elem_size);
                    logit_to_modify_data += (batch_idx * vocab_size + banned_token) * elem_size; 
                    logit_to_modify_stride.data = reinterpret_cast<const int64_t *>(logit_to_modify_data);
                    diopiRequireTensor(ctx, &logit_to_modify, &logit_to_modify_shape, &logit_to_modify_stride, logits_type, diopi_device);
                    
                    diopiScalar_t minus_inf;
                    minus_inf.stype = logits_type;
                    minus_inf.fval = -INFINITY;
                    diopiFill(ctx, logit_to_modify, &minus_inf);
                }
                continue;
            }
        }   
    }
    return diopiSuccess;
}


DIOPI_API diopiError_t diopiCopyD2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atDest = impl::aten::buildATen(dst);
    at::Tensor atSrc = impl::aten::buildATen(src);
    // Set non_blocking true to avoid stream sync thus improving performance.
    // The data is not ready when diopiCopyInp returns.
    // If you need to use it immediately, please call cudaStreamSynchronize first.
    at::native::copy_(atDest, atSrc, async);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief GatherOutput. [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
 * src skip padding in [context_len, max_context_len) and src len <= max_gen_step
 * when src in [max_context_len, ...), dst_idx = src_idx - (max_context_len - context_len)
 * @param[in] ctx The diopi context.
 * @param[out] output_ids : Output ids.shape = [batch_size, max_output_len].type = [int64, int32]
 * @param[in] ids : Ids.shape = [session, batch_size].type = [int64, int32]
 * @param[in] context_lengths : Contextlengths.shape = [batch_size].type = [int64, int32]
 * @param[in] max_context_len : Max context len.type = [int64, int32]
 * @param[in] max_gen_step : Max gen step.type = [int64, int32]
 * @param[in] max_output_len : Max output len.type = [int64, int32]
 * @param[in] batch_size : Batch size.type = [int64, int32]
 */
DIOPI_API diopiError_t diopiGatherOutput(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiConstTensorHandle_t ids,
                                         diopiConstTensorHandle_t context_length, int64_t max_context_len, int64_t max_gen_step, int64_t max_output_len,
                                         int64_t batch_size) {
    diopiDtype_t ids_type, context_length_type;
    diopiGetTensorDtype(ids, &ids_type);
    diopiGetTensorDtype(context_length, &context_length_type);
    if (context_length_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    int64_t ids_elem_size;
    diopiGetTensorElemSize(ids, &ids_elem_size);
    std::cout << "LXZ LOG diopigatheroutput 0" <<std::endl;
    diopiTensorHandle_t context_length_host;
    diopiSize_t context_length_shape;
    diopiGetTensorShape(context_length, &context_length_shape);
    diopiRequireTensor(ctx, &context_length_host, &context_length_shape, nullptr, context_length_type, diopi_host);
    diopiCopyD2H(ctx, context_length_host, context_length, false);
    std::cout << "LXZ LOG diopigatheroutput 0.1" <<std::endl;
    int32_t *context_length_host_data;
    diopiGetTensorData(context_length_host, reinterpret_cast<void **>(&context_length_host_data));
    std::cout << "LXZ LOG diopigatheroutput 0.2" <<std::endl;
    for(int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::cout << "LXZ LOG diopigatheroutput 1" <<std::endl;
        diopiTensorHandle_t src_col;
        diopiSize_t src_col_shape;
        src_col_shape.len = 1;
        src_col_shape.data = &max_gen_step;
        std::cout << "max_gen_step: " << max_gen_step << " max_output_len " << max_output_len << std::endl;
        diopiRequireTensor(ctx, &src_col, &src_col_shape, nullptr, ids_type, diopi_device);
        std::cout << "LXZ LOG GATHEROUTPUT 1.1" <<std::endl;
        diopiSelect(ctx, src_col, ids, 1, batch_idx);
        
        std::cout << "LXZ LOG GATHEROUTPUT 2" <<std::endl;
        diopiTensorHandle_t src_col_front;
        diopiSize_t src_col_front_shape, src_col_front_stride;
        int64_t context_len = static_cast<int64_t>(context_length_host_data[batch_idx]);
        src_col_front_shape.len = 1;
        src_col_front_shape.data = &context_len;
        src_col_front_stride.len = -1;
        char *src_col_front_data;
        diopiGetTensorData(src_col, reinterpret_cast<void **>(&src_col_front_data));
        src_col_front_stride.data = reinterpret_cast<const int64_t *>(src_col_front_data);
        diopiRequireTensor(ctx, &src_col_front, &src_col_front_shape, &src_col_front_stride, ids_type, diopi_device);

        std::cout << "LXZ LOG GATHEROUTPUT 3" <<std::endl;
        diopiTensorHandle_t dst_row_front;
        diopiSize_t dst_row_front_stride;
        dst_row_front_stride.len = -1;
        char* dst_row_front_data;
        diopiGetTensorData(output_ids, reinterpret_cast<void **>(&dst_row_front_data));
        dst_row_front_data += (batch_idx * max_output_len * ids_elem_size);
        dst_row_front_stride.data = reinterpret_cast<const int64_t *>(dst_row_front_data);
        diopiRequireTensor(ctx, &dst_row_front, &src_col_front_shape, &dst_row_front_stride, ids_type, diopi_device);

        std::cout << "LXZ LOG diopigatheroutput 4" <<std::endl;
        diopiCopyD2D(ctx, dst_row_front, src_col_front, false);
        std::cout << "LXZ LOG diopigatheroutput 5" <<std::endl;
        if(max_context_len < max_gen_step) {
            std::cout << "LXZ LOG diopigatheroutput 6" <<std::endl;
            diopiTensorHandle_t src_col_back;
            diopiSize_t src_col_back_shape, src_col_back_stride;
            src_col_front_shape.len = 1;
            int64_t back_len = max_gen_step - max_context_len;
            src_col_front_shape.data = &back_len;
            src_col_front_stride.len = -1;
            char *src_col_back_data;
            diopiGetTensorData(src_col, reinterpret_cast<void **>(&src_col_back_data));
            src_col_back_data += (max_context_len * ids_elem_size);
            src_col_back_stride.data = reinterpret_cast<const int64_t *>(src_col_back_data);
            diopiRequireTensor(ctx, &src_col_back, &src_col_back_shape, &src_col_back_stride, ids_type, diopi_device);
            std::cout << "LXZ LOG diopigatheroutput 7" <<std::endl;
            diopiTensorHandle_t dst_row_back;
            diopiSize_t dst_row_back_stride;
            dst_row_back_stride.len = -1;
            char* dst_row_back_data;
            diopiGetTensorData(output_ids, reinterpret_cast<void **>(&dst_row_back_data));
            dst_row_back_data += ((batch_idx * max_output_len + context_len) * ids_elem_size);
            diopiRequireTensor(ctx, &dst_row_back, &src_col_back_shape, &dst_row_back_stride, ids_type, diopi_device);
            std::cout << "LXZ LOG diopigatheroutput 8" <<std::endl;
            diopiCopyD2D(ctx, dst_row_back, src_col_back, false);
            std::cout << "LXZ LOG diopigatheroutput 9" <<std::endl;
        }
    }
    return diopiSuccess;
}

}  // extern "C"
