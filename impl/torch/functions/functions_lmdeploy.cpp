/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>
#include <math.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/torch.h>

#include <cstring>

#ifdef USE_HIP
#include <miopen/version.h>
#endif

#define FLT_MIN __FLT_MIN__
#define FLT_MAX __FLT_MAX__

#include "../context.h"
#include "../helper.hpp"
#include "../vision_kernel.h"

extern "C" {

#define DIOPI_CHECK(expr)                                           \
    do {                                                            \
        diopiError_t ret = expr;                                    \
        if (ret != diopiSuccess) {                                  \
            printf(#expr " error at %s:%d.\n", __FILE__, __LINE__); \
            return ret;                                             \
        }                                                           \
    } while (false);

#define DIOPI_CHECK_FMT(expr, fmt, args...)                          \
    do {                                                             \
        diopiError_t ret = expr;                                     \
        if (ret != diopiSuccess) {                                   \
            printf(#fmt " at %s:%d.\n", ##args, __FILE__, __LINE__); \
            return ret;                                              \
        }                                                            \
    } while (false);

DIOPI_API diopiError_t diopiLmdeployCopyH2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_device || src_dev != diopiDevice_t::diopi_host) {
        return diopiErrorOccurred;
    }

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

DIOPI_API diopiError_t diopiLmdeployCopyD2H(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_host || src_dev != diopiDevice_t::diopi_device) {
        return diopiErrorOccurred;
    }

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

DIOPI_API diopiError_t diopiLmdeployCopyD2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_device || src_dev != diopiDevice_t::diopi_device) {
        return diopiErrorOccurred;
    }

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

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input) {
    diopiDtype_t dtype;
    DIOPI_CHECK(diopiGetTensorDtype(input, &dtype));
    diopiSize_t shape;
    DIOPI_CHECK(diopiGetTensorShape(input, &shape));
    diopiDevice_t device;
    DIOPI_CHECK(diopiGetTensorDevice(input, &device));

    DIOPI_CHECK(diopiRequireTensor(ctx, out, &shape, nullptr, dtype, device));
    return diopiSuccess;
}

int64_t *getDataOffsetPtr(diopiTensorHandle_t tensor, int64_t offset) {
    char *ptr = nullptr;
    diopiGetTensorData(tensor, reinterpret_cast<void **>(&ptr));
    if (offset == 0) {
        return reinterpret_cast<int64_t *>(ptr);
    }
    int64_t elem_size;
    diopiGetTensorElemSize(tensor, &elem_size);
    return reinterpret_cast<int64_t *>(ptr + offset * elem_size);
}

DIOPI_API diopiError_t diopiFusedSiluFfnInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t weight1,
                                            diopiConstTensorHandle_t weight2, diopiConstTensorHandle_t weight3, diopiTensorHandle_t workspace,
                                            int64_t *workspace_size, int64_t fusion_level) {
    if (fusion_level >= 0) {
        diopiSize_t shapeinfo;
        diopiGetTensorShape(inoutput, &shapeinfo);
        int64_t token_num = shapeinfo.data[0];
        diopiGetTensorShape(weight1, &shapeinfo);
        int64_t inter_size = shapeinfo.data[1];
        int64_t itemsize = -1;
        diopiGetTensorElemSize(inoutput, &itemsize);
        if (*workspace_size < 0) {
            *workspace_size = 2 * itemsize * token_num * inter_size;
            return diopiSuccess;
        }
        void *dataptr;
        diopiGetTensorData(workspace, &dataptr);
        diopiDevice_t device;
        diopiGetTensorDevice(workspace, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(workspace, &dtype);
        std::vector<int64_t> shape(2);
        diopiSize_t newshape{shape.data(), 2};
        shape[0] = token_num;
        shape[1] = inter_size;
        diopiSize_t strideW1{static_cast<const int64_t *>(reinterpret_cast<int64_t *>(dataptr)), -1};
        diopiSize_t strideW3{static_cast<const int64_t *>(reinterpret_cast<int64_t *>(reinterpret_cast<char *>(dataptr) + itemsize * token_num * inter_size)),
                             -1};
        diopiTensorHandle_t matmulW1;
        diopiTensorHandle_t matmulW3;
        diopiRequireTensor(ctx, &matmulW1, &newshape, &strideW1, dtype, device);
        diopiRequireTensor(ctx, &matmulW3, &newshape, &strideW3, dtype, device);

        DIOPI_CHECK(diopiMm(ctx, matmulW1, inoutput, weight1));
        DIOPI_CHECK(diopiMm(ctx, matmulW3, inoutput, weight3));
        DIOPI_CHECK(diopiSiluInp(ctx, matmulW1));
        DIOPI_CHECK(diopiMulInp(ctx, matmulW1, matmulW3));
        DIOPI_CHECK(diopiMm(ctx, inoutput, matmulW1, weight2));
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}

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

    if (total_padding_count == nullptr) {
        diopiRequireTensor(ctx, &total_padding_count, &in_shape, &in_stride, in_type, in_device);
    }

    diopiFill(ctx, total_padding_count, &max_input_length_scalar);

    diopiScalar_t one;
    one.stype = diopi_dtype_int64;
    one.ival = 1;
    diopiSubInp(ctx, total_padding_count, const_cast<diopiConstTensorHandle_t>(input_lengths), &one);
    return diopiSuccess;
}

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
    diopiSize_t in_shape;
    diopiDevice_t in_device;
    diopiGetTensorDtype(finished, &in_type);
    diopiGetTensorShape(finished, &in_shape);
    diopiGetTensorDevice(finished, &in_device);
    diopiTensorHandle_t finished_fp64;
    diopiRequireTensor(ctx, &finished_fp64, &in_shape, nullptr, diopi_dtype_float64, in_device);
    diopiCastDtype(ctx, finished_fp64, finished);

    diopiGetTensorShape(finished_sum, &in_shape);
    diopiGetTensorDevice(finished_sum, &in_device);
    diopiTensorHandle_t finished_sum_fp64;
    diopiRequireTensor(ctx, &finished_sum_fp64, &in_shape, nullptr, diopi_dtype_float64, diopi_device);
    diopiCastDtype(ctx, finished_sum_fp64, finished_sum);

    diopiSize_t dim_zero;
    int64_t tmp_zero = 0;
    dim_zero.data = &tmp_zero;
    dim_zero.len = 1;
    diopiSum(ctx, finished_sum_fp64, finished_fp64, dim_zero);

    diopiCastDtype(ctx, finished_sum, finished_sum_fp64);

    diopiGetTensorDtype(finished, &in_type);
    diopiGetTensorShape(finished, &in_shape);
    diopiGetTensorDevice(finished, &in_device);
    diopiTensorHandle_t h_finished;
    diopiRequireTensor(ctx, &h_finished, &in_shape, nullptr, in_type, diopi_host);
    diopiCopyD2H(ctx, h_finished, finished, false);
    diopiAll(ctx, should_stop, h_finished, &tmp_zero);
    return diopiSuccess;
}

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
    diopiSize_t this_step_ids_shape;
    this_step_ids_shape.len = 1;
    this_step_ids_shape.data = &batch_size;

    diopiRequireTensor(ctx, &this_step_ids, &this_step_ids_shape, nullptr, in_type, in_device);

    diopiSelect(ctx, this_step_ids, all_ids, 0, step);
    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, this_step_ids);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiInputIdsEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t input_ids,
                                                               diopiConstTensorHandle_t embedding_table, const int64_t input_lengths,
                                                               const int64_t hidden_units) {
    if (from_tensor == nullptr || input_ids == nullptr || embedding_table == nullptr) {
        return diopiErrorOccurred;
    }

    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, input_ids);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiStopWordsCriterion(diopiContextHandle_t ctx, diopiConstTensorHandle_t output_ids, diopiConstTensorHandle_t stop_words,
                                               diopiTensorHandle_t finished, int64_t id_offset, int64_t stop_words_len, int64_t batch_size, int64_t step) {
    std::cout << "LXZ LOG" << std::endl;
    // always id_offset = 0
    if (output_ids == nullptr || stop_words == nullptr || finished == nullptr) {
        return diopiErrorOccurred;
    }

    diopiTensorHandle_t stop_words_host;
    diopiSize_t stop_words_shape;
    diopiDtype_t stop_words_type;
    diopiGetTensorDtype(stop_words, &stop_words_type);
    if (stop_words_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    diopiGetTensorShape(stop_words, &stop_words_shape);
    diopiRequireTensor(ctx, &stop_words_host, &stop_words_shape, nullptr, diopi_dtype_int32, diopi_host);
    diopiCopyD2H(ctx, stop_words_host, stop_words, false);

    const int32_t *stop_words_ptr;
    int32_t *stop_words_host_ptr;

    int64_t finished_elem_size;
    diopiGetTensorElemSize(finished, &finished_elem_size);
    diopiDtype_t finished_type;
    diopiGetTensorDtype(finished, &finished_type);
    assert(finished_type == diopi_dtype_bool);

    diopiGetTensorDataConst(stop_words, reinterpret_cast<const void **>(&stop_words_ptr));
    diopiGetTensorData(stop_words_host, reinterpret_cast<void **>(&stop_words_host_ptr));

    diopiDtype_t ids_type;
    diopiGetTensorDtype(output_ids, &ids_type);
    if (ids_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t *base_stop_words_host = stop_words_host_ptr + batch_idx * 2 * stop_words_len;
        const int32_t *base_offsets_host = base_stop_words_host + stop_words_len;
        const int32_t *base_stop_word = stop_words_ptr + batch_idx * 2 * stop_words_len;

        for (int64_t stop_word_idx = 0; stop_word_idx < stop_words_len; ++stop_word_idx) {
            if (base_stop_words_host[stop_word_idx] < 0) {
                continue;
            }
            const int32_t stop_word_start_idx = (stop_word_idx > 0) ? base_offsets_host[stop_word_idx - 1] : 0;
            const int32_t stop_word_end_idx = base_offsets_host[stop_word_idx] - 1;
            const int64_t stop_word_len = stop_word_end_idx - stop_word_start_idx + 1;

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
            int64_t output_ids_col_shape_tmp = step + 1;
            output_ids_col_shape.data = &output_ids_col_shape_tmp;
            diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, ids_type, diopi_device);
            diopiSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

            diopiTensorHandle_t output_ids_to_compare;
            char *output_ids_to_compare_data;
            diopiGetTensorData(output_ids_col, reinterpret_cast<void **>(&output_ids_to_compare_data));
            int64_t elem_size;
            diopiGetTensorElemSize(output_ids_col, &elem_size);
            output_ids_to_compare_data += (step - stop_word_len + 1) * elem_size;
            stride.len = -1;
            stride.data = reinterpret_cast<const int64_t *>(reinterpret_cast<int64_t *>(output_ids_to_compare_data));
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
            bool *cmp_res_sum_host_data;
            diopiGetTensorData(cmp_res_sum_host, reinterpret_cast<void **>(&cmp_res_sum_host_data));

            if (cmp_res_sum_host_data[0]) {
                diopiScalar_t true_scalar;
                true_scalar.stype = diopi_dtype_bool;
                true_scalar.ival = 1;

                diopiTensorHandle_t finished_to_modify;
                diopiSize_t finished_to_modify_stride;
                finished_to_modify_stride.len = -1;
                finished_to_modify_stride.data = getDataOffsetPtr(finished, batch_idx);

                diopiRequireTensor(ctx, &finished_to_modify, &cmp_res_sum_shape, &finished_to_modify_stride, diopi_dtype_bool, diopi_device);

                diopiFill(ctx, finished_to_modify, &true_scalar);

                break;
            }
        }
    }
    return diopiSuccess;
}

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
    int32_t *bad_words_host_ptr;

    diopiGetTensorDataConst(bad_words, reinterpret_cast<const void **>(&bad_words_ptr));
    diopiGetTensorData(bad_words_host, reinterpret_cast<void **>(&bad_words_host_ptr));

    diopiDtype_t ids_type;
    diopiGetTensorDtype(output_ids, &ids_type);
    if (ids_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t *base_bad_words_host = share_words ? bad_words_host_ptr : bad_words_host_ptr + batch_idx * 2 * bad_words_len;
        const int32_t *base_offsets_host = base_bad_words_host + bad_words_len;
        const int32_t *base_bad_word = share_words ? bad_words_ptr : bad_words_ptr + batch_idx * 2 * bad_words_len;

        for (int64_t bad_word_idx = 0; bad_word_idx < bad_words_len; ++bad_word_idx) {
            if (base_bad_words_host[bad_word_idx] < 0) {
                continue;
            }
            const int32_t bad_word_start_idx = (bad_word_idx > 0) ? base_offsets_host[bad_word_idx - 1] : 0;
            const int32_t bad_word_end_idx = base_offsets_host[bad_word_idx] - 1;
            const int64_t bad_word_len = bad_word_end_idx - bad_word_start_idx + 1;

            if (step + 1 < bad_word_len) {
                continue;
            }

            bool *cmp_res_sum_host_data = nullptr;
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
                    char *logit_to_modify_data;
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

    diopiTensorHandle_t context_length_host;
    diopiSize_t context_length_shape;
    diopiGetTensorShape(context_length, &context_length_shape);
    diopiRequireTensor(ctx, &context_length_host, &context_length_shape, nullptr, context_length_type, diopi_host);
    diopiCopyD2H(ctx, context_length_host, context_length, false);

    int32_t *context_length_host_data;
    diopiGetTensorData(context_length_host, reinterpret_cast<void **>(&context_length_host_data));

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        diopiTensorHandle_t src_col;
        diopiSize_t src_col_shape;
        src_col_shape.len = 1;
        src_col_shape.data = &max_output_len;

        diopiRequireTensor(ctx, &src_col, &src_col_shape, nullptr, ids_type, diopi_device);
        diopiSelect(ctx, src_col, ids, 1, batch_idx);

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

        diopiTensorHandle_t dst_row_front;
        diopiSize_t dst_row_front_stride;
        dst_row_front_stride.len = -1;
        char *dst_row_front_data;
        diopiGetTensorData(output_ids, reinterpret_cast<void **>(&dst_row_front_data));
        dst_row_front_data += (batch_idx * max_output_len * ids_elem_size);
        dst_row_front_stride.data = reinterpret_cast<const int64_t *>(dst_row_front_data);
        diopiRequireTensor(ctx, &dst_row_front, &src_col_front_shape, &dst_row_front_stride, ids_type, diopi_device);

        diopiCopyD2D(ctx, dst_row_front, src_col_front, false);
        if (max_context_len < max_gen_step) {
            diopiTensorHandle_t src_col_back;
            diopiSize_t src_col_back_shape, src_col_back_stride;
            src_col_back_shape.len = 1;
            int64_t back_len = max_gen_step - max_context_len;
            src_col_back_shape.data = &back_len;
            src_col_back_stride.len = -1;
            char *src_col_back_data;
            diopiGetTensorData(src_col, reinterpret_cast<void **>(&src_col_back_data));
            src_col_back_data += (max_context_len * ids_elem_size);
            src_col_back_stride.data = reinterpret_cast<const int64_t *>(src_col_back_data);
            diopiRequireTensor(ctx, &src_col_back, &src_col_back_shape, &src_col_back_stride, ids_type, diopi_device);
            diopiTensorHandle_t dst_row_back;
            diopiSize_t dst_row_back_stride;
            dst_row_back_stride.len = -1;
            char *dst_row_back_data;
            diopiGetTensorData(output_ids, reinterpret_cast<void **>(&dst_row_back_data));
            dst_row_back_data += ((batch_idx * max_output_len + context_len) * ids_elem_size);
            dst_row_back_stride.data = reinterpret_cast<const int64_t *>(dst_row_back_data);
            diopiRequireTensor(ctx, &dst_row_back, &src_col_back_shape, &dst_row_back_stride, ids_type, diopi_device);
            diopiCopyD2D(ctx, dst_row_back, src_col_back, false);
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchApplyRepetitionPenaltyInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t penalties,
                                                           diopiConstTensorHandle_t output_ids, const int64_t batch_size, const int64_t vocab_size,
                                                           diopiConstTensorHandle_t input_lengths, const int64_t max_input_length, const int64_t step,
                                                           const int64_t penalty_type) {
    if (logits == nullptr || penalties == nullptr || output_ids == nullptr || input_lengths == nullptr) {
        return diopiErrorOccurred;
    }

    if (penalty_type != 0 && penalty_type != 1 && penalty_type != 2) {
        return diopiErrorOccurred;
    }

    if (penalty_type == 0) {
        return diopiSuccess;
    }

    diopiDtype_t logits_type, penalties_type, input_lengths_type, output_ids_type;
    diopiGetTensorDtype(logits, &logits_type);
    diopiGetTensorDtype(penalties, &penalties_type);
    diopiGetTensorDtype(input_lengths, &input_lengths_type);
    diopiGetTensorDtype(output_ids, &output_ids_type);

    if (input_lengths_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    int64_t logits_elem_size, penalties_elem_size, input_lengths_elem_size, output_ids_elem_size;
    diopiGetTensorElemSize(logits, &logits_elem_size);
    diopiGetTensorElemSize(penalties, &penalties_elem_size);
    diopiGetTensorElemSize(input_lengths, &input_lengths_elem_size);
    diopiGetTensorElemSize(output_ids, &output_ids_elem_size);

    int32_t *input_lengths_host_data = nullptr;
    if (input_lengths != nullptr) {
        diopiTensorHandle_t input_lengths_host;
        diopiSize_t input_lengths_shape;
        diopiGetTensorShape(input_lengths, &input_lengths_shape);
        diopiRequireTensor(ctx, &input_lengths_host, &input_lengths_shape, nullptr, input_lengths_type, diopi_host);

        diopiCopyD2H(ctx, input_lengths_host, input_lengths, false);
        diopiGetTensorData(input_lengths_host, reinterpret_cast<void **>(&input_lengths_host_data));
    }

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        diopiTensorHandle_t output_ids_col;
        diopiSize_t output_ids_col_shape;
        output_ids_col_shape.len = 1;
        output_ids_col_shape.data = &step;
        diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, output_ids_type, diopi_device);

        diopiSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

        diopiTensorHandle_t output_ids_col_front;
        diopiSize_t output_ids_col_front_shape, output_ids_col_front_stride;

        output_ids_col_front_shape.len = 1;
        int64_t input_len = input_lengths_host_data == nullptr ? max_input_length : static_cast<int64_t>(input_lengths_host_data[batch_idx]);
        output_ids_col_front_shape.data = &input_len;

        output_ids_col_front_stride.len = -1;
        char *output_ids_col_front_data;
        diopiGetTensorData(output_ids_col, reinterpret_cast<void **>(&output_ids_col_front_data));
        output_ids_col_front_stride.data = reinterpret_cast<const int64_t *>(output_ids_col_front_data);

        diopiRequireTensor(ctx, &output_ids_col_front, &output_ids_col_front_shape, &output_ids_col_front_stride, output_ids_type, diopi_device);
        diopiTensorHandle_t valid_output_ids_col = output_ids_col_front;
        int64_t vaild_output_ids_col_len = input_len;

        if (max_input_length < step) {
            diopiTensorHandle_t output_ids_col_back;
            diopiSize_t output_ids_col_back_shape, output_ids_col_back_stride;
            output_ids_col_back_shape.len = 1;
            int64_t back_len = step - max_input_length;
            output_ids_col_back_shape.data = &back_len;
            output_ids_col_back_stride.len = -1;
            char *output_ids_col_back_data;
            diopiGetTensorData(output_ids_col, reinterpret_cast<void **>(&output_ids_col_back_data));
            output_ids_col_back_data += (max_input_length * output_ids_elem_size);
            output_ids_col_back_stride.data = reinterpret_cast<const int64_t *>(output_ids_col_back_data);
            diopiRequireTensor(ctx, &output_ids_col_back, &output_ids_col_back_shape, &output_ids_col_back_stride, output_ids_type, diopi_device);

            vaild_output_ids_col_len = input_len + back_len;
            diopiSize_t valid_output_ids_col_shape;
            valid_output_ids_col_shape.len = 1;
            valid_output_ids_col_shape.data = &vaild_output_ids_col_len;
            diopiRequireTensor(ctx, &valid_output_ids_col, &valid_output_ids_col_shape, nullptr, output_ids_type, diopi_device);
            diopiConstTensorHandle_t to_cat[2] = {output_ids_col_front, output_ids_col_back};
            diopiCat(ctx, valid_output_ids_col, to_cat, 2, 0);
        }

        diopiTensorHandle_t logits_this_batch;
        diopiSize_t logits_this_batch_shape, logits_this_batch_stride;
        logits_this_batch_shape.len = 1;
        logits_this_batch_shape.data = &vocab_size;
        logits_this_batch_stride.len = -1;
        char *logits_this_batch_data;
        diopiGetTensorData(logits, reinterpret_cast<void **>(&logits_this_batch_data));
        logits_this_batch_data += (batch_idx * vocab_size * logits_elem_size);
        logits_this_batch_stride.data = reinterpret_cast<const int64_t *>(logits_this_batch_data);
        diopiRequireTensor(ctx, &logits_this_batch, &logits_this_batch_shape, &logits_this_batch_stride, logits_type, diopi_device);

        diopiTensorHandle_t logits_to_penalize;
        diopiSize_t logits_to_penalize_shape;
        logits_to_penalize_shape.len = 1;
        logits_to_penalize_shape.data = &vaild_output_ids_col_len;
        diopiRequireTensor(ctx, &logits_to_penalize, &logits_to_penalize_shape, nullptr, logits_type, diopi_device);
        diopiIndexSelect(ctx, logits_to_penalize, logits_this_batch, 0, valid_output_ids_col);

        diopiTensorHandle_t penalties_this_batch;
        diopiSize_t penalties_this_batch_shape, penalties_this_batch_stride;
        penalties_this_batch_shape.len = 1;
        int64_t tmp_one = 1;
        penalties_this_batch_shape.data = &tmp_one;

        penalties_this_batch_stride.len = -1;
        const char *penalties_this_batch_data;
        diopiGetTensorDataConst(penalties, reinterpret_cast<const void **>(&penalties_this_batch_data));
        penalties_this_batch_data += (batch_idx * penalties_elem_size);
        penalties_this_batch_stride.data = reinterpret_cast<const int64_t *>(penalties_this_batch_data);
        diopiRequireTensor(ctx, &penalties_this_batch, &penalties_this_batch_shape, &penalties_this_batch_stride, penalties_type, diopi_device);

        diopiTensorHandle_t penalties_this_batch_expand;
        diopiSize_t penalties_this_batch_expand_shape;
        penalties_this_batch_expand_shape.len = 1;
        penalties_this_batch_expand_shape.data = &vaild_output_ids_col_len;
        diopiRequireTensor(ctx, &penalties_this_batch_expand, &penalties_this_batch_expand_shape, nullptr, penalties_type, diopi_device);
        diopiExpand(ctx, penalties_this_batch_expand, penalties_this_batch);
        diopiScalar_t one_scalar;
        one_scalar.stype = diopi_dtype_int64;
        one_scalar.ival = 1;

        if (penalty_type == 1) {
            diopiSubInp(ctx, logits_to_penalize, penalties_this_batch_expand, &one_scalar);
        } else {
            diopiTensorHandle_t ge_zero_mask, lt_zero_mask;
            diopiSize_t ge_than_zero_mask_shape;
            ge_than_zero_mask_shape.len = 1;
            ge_than_zero_mask_shape.data = &vaild_output_ids_col_len;
            diopiRequireTensor(ctx, &ge_zero_mask, &ge_than_zero_mask_shape, nullptr, diopi_dtype_bool, diopi_device);
            diopiRequireTensor(ctx, &lt_zero_mask, &ge_than_zero_mask_shape, nullptr, diopi_dtype_bool, diopi_device);

            diopiScalar_t zero_scalar;
            zero_scalar.stype = diopi_dtype_float32;
            zero_scalar.ival = 0.0f;
            diopiTensorHandle_t mul_penalty, div_penalty;
            diopiRequireTensor(ctx, &mul_penalty, &logits_to_penalize_shape, nullptr, logits_type, diopi_device);
            diopiRequireTensor(ctx, &div_penalty, &logits_to_penalize_shape, nullptr, logits_type, diopi_device);

            diopiClampScalar(ctx, mul_penalty, logits_to_penalize, nullptr, &zero_scalar);
            diopiClampScalar(ctx, div_penalty, logits_to_penalize, &zero_scalar, nullptr);
            diopiMulInp(ctx, mul_penalty, penalties_this_batch_expand);
            diopiDivInp(ctx, div_penalty, penalties_this_batch_expand, RoundModeNone);
            diopiAdd(ctx, logits_to_penalize, mul_penalty, div_penalty, &one_scalar);
        }
        char reduce[0];
        diopiTensorHandle_t scatter_index = valid_output_ids_col;

        if (output_ids_type != diopi_dtype_int64) {
            diopiSize_t scatter_index_shape;
            diopiGetTensorShape(valid_output_ids_col, &scatter_index_shape);
            diopiRequireTensor(ctx, &scatter_index, &scatter_index_shape, nullptr, diopi_dtype_int64, diopi_device);
            diopiCastDtype(ctx, scatter_index, valid_output_ids_col);
        }

        diopiScatterInp(ctx, logits_this_batch, 0, logits_to_penalize, scatter_index, reduce);
    }

    return diopiSuccess;
}

diopiError_t diopiBatchApplyTemperaturePenaltyInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t bias,
                                                  diopiConstTensorHandle_t temperatures, const int64_t batch_size, const int64_t vocab_size,
                                                  const int64_t vocab_size_padd) {
    assert(vocab_size_padd >= vocab_size);
    assert(logits != nullptr);
    assert(temperatures != nullptr);
    diopiDtype_t logits_dtype;
    DIOPI_CHECK(diopiGetTensorDtype(logits, &logits_dtype));
    diopiSize_t logits_shape;
    DIOPI_CHECK(diopiGetTensorShape(logits, &logits_shape));
    assert(logits_shape.len == 2 && logits_shape.data[0] == batch_size && logits_shape[1] == vocab_size_padded);
    diopiTensorHandle_t lhs;
    std::vector<int64_t> lhs_shape_vec = {batch_size, vocab_size};
    diopiSize_t lhs_shape{lhs_shape_vec.data(), 2};
    DIOPI_CHECK(diopiRequireTensor(ctx, &lhs, &lhs_shape, nullptr, logits_dtype, diopi_device));
    DIOPI_CHECK(diopiSlice(ctx, lhs, logits, 1, 0, vocab_size, 1));

    diopiTensorHandle_t rhs = nullptr;
    if (vocab_size_padd > vocab_size) {
        std::vector<int64_t> rhs_shape_vec = {batch_size, vocab_size_padd - vocab_size};
        diopiSize_t rhs_shape{rhs_shape_vec.data(), 2};
        DIOPI_CHECK(diopiRequireTensor(ctx, &rhs, &rhs_shape, nullptr, logits_dtype, diopi_device));
        DIOPI_CHECK(diopiSlice(ctx, rhs, logits, 1, vocab_size, vocab_size_padd, 1));
        double MAX_T_VAL = (logits_dtype == diopiDtype_t::diopi_dtype_float16 ? 65504.F : FLT_MAX);
        diopiScalar_t scalar_val;
        scalar_val.stype = logits_dtype;
        scalar_val.fval = -MAX_T_VAL;
        DIOPI_CHECK(diopiFill(ctx, rhs, &scalar_val));
    }
    diopiTensorHandle_t new_temperatures = nullptr;
    DIOPI_CHECK(makeTensorLike(ctx, &new_temperatures, temperatures));
    diopiDtype_t temperatures_dtype;
    DIOPI_CHECK(diopiGetTensorDtype(temperatures, &temperatures_dtype));

    assert(temperatures_dtype == diopi_dtype_float32);

    diopiScalar_t eps_scalar;
    eps_scalar.stype = temperatures_dtype;
    eps_scalar.fval = 1e-6;
    diopiScalar_t one_scalar;
    one_scalar.stype = temperatures_dtype;
    one_scalar.fval = 1.0;
    DIOPI_CHECK(diopiAddScalar(ctx, new_temperatures, temperatures, &eps_scalar, &one_scalar));

    if (bias != nullptr) {
        diopiScalar_t t;
        t.stype = logits_dtype;
        t.fval = 1.0;
        DIOPI_CHECK(diopiAddInp(ctx, lhs, bias, &t));
    }

    diopiSize_t new_temperatures_shape;
    DIOPI_CHECK(diopiGetTensorShape(new_temperatures, &new_temperatures_shape));
    diopiTensorHandle_t new_temperatures_host;
    DIOPI_CHECK(diopiRequireTensor(ctx, &new_temperatures_host, &new_temperatures_shape, nullptr, temperatures_dtype, diopi_host));
    DIOPI_CHECK(diopiCopyD2H(ctx, new_temperatures_host, new_temperatures, false));
    char *new_temperatures_host_data;

    DIOPI_CHECK(diopiGetTensorData(new_temperatures_host, reinterpret_cast<void **>(&new_temperatures_host_data)));
    for (int64_t i = 0; i < batch_size; ++i) {
        diopiScalar_t temperature_scalar;
        temperature_scalar.stype = diopi_dtype_float32;
        temperature_scalar.fval = reinterpret_cast<float *>(new_temperatures_host_data)[i];

        diopiTensorHandle_t logits_row;
        diopiSize_t logits_row_shape, logits_row_stride;
        logits_row_shape.len = 1;
        logits_row_shape.data = &vocab_size;

        logits_row_stride.len = -1;
        logits_row_stride.data = getDataOffsetPtr(logits, i * vocab_size);
        DIOPI_CHECK(diopiRequireTensor(ctx, &logits_row, &logits_row_shape, &logits_row_stride, logits_dtype, diopi_device));
        DIOPI_CHECK(diopiDivInpScalar(ctx, logits_row, &temperature_scalar, RoundModeNone));
    }

    if (rhs == nullptr) {
        DIOPI_CHECK(diopiCopyInp(ctx, lhs, logits));
    } else {
        std::array<diopiConstTensorHandle_t, 2> tensors = {lhs, rhs};
        DIOPI_CHECK(diopiCat(ctx, logits, tensors.data(), tensors.size(), 1));
    }
    return diopiSuccess;
}

}  // extern "C"
