/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>
#include <float.h>
#include <stdio.h>

#include <array>
#include <cassert>
#include <iostream>
#include <vector>

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

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    diopiDtype_t dtype;
    DIOPI_CHECK(diopiGetTensorDtype(input, &dtype));
    diopiSize_t shape;
    DIOPI_CHECK(diopiGetTensorShape(input, &shape));
    diopiDevice_t device;
    DIOPI_CHECK(diopiGetTensorDevice(input, &device));

    DIOPI_CHECK(diopiRequireTensor(ctx, out, &shape, nullptr, dtype, device));
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
    std::vector<int64_t> lhs_shape_vec(batch_size, vocab_size);
    diopiSize_t lhs_shape{lhs_shape_vec.data(), 2};
    DIOPI_CHECK(diopiRequireTensor(ctx, &lhs, &lhs_shape, nullptr, logits_dtype, diopi_device));
    DIOPI_CHECK(diopiSlice(ctx, lhs, logits, 1, 0, vocab_size, 1));

    diopiTensorHandle_t rhs = nullptr;
    if (vocab_size_padd > vocab_size) {
        std::vector<int64_t> rhs_shape_vec(batch_size, vocab_size_padd - vocab_size);
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
    diopiScalar_t eps_scalar;
    eps_scalar.stype = temperatures_dtype;
    eps_scalar.fval = 1e-6;
    DIOPI_CHECK(diopiAddInp(ctx, new_temperatures, temperatures, &eps_scalar));

    if (bias != nullptr) {
        diopiScalar_t t;
        t.stype = logits_dtype;
        t.fval = 1.0;
        DIOPI_CHECK(diopiAddInp(ctx, lhs, bias, &t));
    }
    DIOPI_CHECK(diopiDivInp(ctx, lhs, new_temperatures, RoundModeNone));

    if (rhs == nullptr) {
        DIOPI_CHECK(diopiCopyInp(ctx, lhs, logits));
    } else {
        std::array<diopiConstTensorHandle_t, 2> tensors = {lhs, rhs};
        DIOPI_CHECK(diopiCat(ctx, logits, tensors.data(), tensors.size(), 1));
    }
    return diopiSuccess;
}

}  // extern "C"
