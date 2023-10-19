/**************************************************************************************************
 * Copyright 2022 Enflame. All Rights Reserved.
 * License: BSD 3-Clause
 * Author: boris.wu
 *
 *************************************************************************************************/
#include <diopi/diopirt.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include "log.h"
#include "ops.h"

namespace impl {
namespace topsrider {

class TopsOpInit {
public:
    ~TopsOpInit() { impl::tops::topsLibInit(); }

    ~TopsOpInit() { impl::tops::topsLibFinalize(); }
};

static TopsOpInit topsop_init;

static const char *name = "GcuDevice";
static char version[1024] = {0};

const char *diopiGetVendorName() { return name; }

const char *diopiGetImplVersion() {
    int rt_version = 2100;
    if (strlen(version) == 0) {
        const char *diopiVersion = diopiGetVersion();
        sprintf(version, "TopsRt Version: %d; %s", rt_version, diopiVersion);
    }
    return version;
}

const char *diopiGetLastErrorString() { return tops_get_last_error_string(); }

DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *start, const diopiScalar_t *end,
                                   const diopiScalar_t *step) {
    TOPSOP_LOG();
    return impl::tops::topsArange(ctx, out, start, end, step);
}

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t *alpha_value) {
    TOPSOP_LOG();
    return impl::tops::topsAdd(ctx, out, input, other, alpha_value);
}

DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                      const diopiScalar_t *alpha) {
    TOPSOP_LOG();
    return impl::tops::topsAddScalar(ctx, out, input, other, alpha);
}
DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t *alpha_value) {
    TOPSOP_LOG();
    return impl::tops::topsSub(ctx, out, input, other, alpha_value);
}
DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                      const diopiScalar_t *alpha) {
    TOPSOP_LOG();
    return impl::tops::topsSubScalar(ctx, out, input, other, alpha);
}
DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsMul(ctx, out, input, other);
}
DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsMulScalar(ctx, out, input, other);
}
DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t rounding_mode) {
    TOPSOP_LOG();
    return impl::tops::topsDiv(ctx, out, input, other, rounding_mode);
}
DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                      diopiRoundMode_t rounding_mode) {
    TOPSOP_LOG();
    return impl::tops::topsDivScalar(ctx, out, input, other, rounding_mode);
}

DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    TOPSOP_LOG();
    return impl::tops::topsPowTensor(ctx, out, input, exponent);
}

DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *input, diopiConstTensorHandle_t exponent) {
    TOPSOP_LOG();
    return impl::tops::topsPowScalar(ctx, out, input, exponent);
}

DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *exponent) {
    TOPSOP_LOG();
    return impl::tops::topsPow(ctx, out, input, exponent);
}

DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t bias, diopiSize_t stride_, diopiSize_t padding_, diopiSize_t dilation_, int64_t groups) {
    TOPSOP_LOG();
    return impl::tops::topsConvolution2d(ctx, out, input, weight, bias, stride_, padding_, dilation_, groups);
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsSoftmax(ctx, out, input, dim, diopi_dtype_float32);
}

DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsSigmoid(ctx, out, input);
}

DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    TOPSOP_LOG();
    return impl::tops::topsBCELoss(ctx, out, input, target, weight, reduction);
}

DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction) {
    TOPSOP_LOG();
    return impl::tops::topsBCELossBackward(ctx, grad_input, grad_output, input, target, weight, reduction);
}

DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    TOPSOP_LOG();
    return impl::tops::topsBatchNorm(ctx, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
}

DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    TOPSOP_LOG();
    return impl::tops::topsBCEWithLogits(ctx, out, input, target, weight, pos_weight, reduction);
}

DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *value) {
    TOPSOP_LOG();
    return impl::tops::topsFill(ctx, input, value);
}
DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t *tensors, int64_t num_inputs, int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsCat(ctx, out, tensors, num_inputs, dim);
}

DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsRelu(ctx, out, input);
}
DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsReluInp(ctx, input);
}

DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *negative_slope) {
    TOPSOP_LOG();
    return impl::tops::topsLeakyRelu(ctx, out, input, negative_slope);
}
DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *negative_slope) {
    TOPSOP_LOG();
    return impl::tops::topsLeakyReluInp(ctx, input, negative_slope);
}
DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t *negative_slope, bool input_is_result) {
    TOPSOP_LOG();
    return impl::tops::topsLeakyReluBackward(ctx, grad_input, grad_output, input, negative_slope, input_is_result);
}

DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiReduction_t reduction) {
    TOPSOP_LOG();
    return impl::tops::topsMSELoss(ctx, out, input, target, reduction);
}
DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    TOPSOP_LOG();
    return impl::tops::topsMSELossBackward(ctx, grad_input, grad_output, input, target, reduction);
}

DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    TOPSOP_LOG();
    return impl::tops::topsDivInp(ctx, input, other, rounding_mode);
}

DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other, diopiRoundMode_t rounding_mode) {
    TOPSOP_LOG();
    return impl::tops::topsDivScalarInp(ctx, input, other, rounding_mode);
}

DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsEq(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsEq(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsEqScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsEqScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsGe(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsGe(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsGeScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsGt(ctx, out, input, other);
}
DIOPI_API diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsGtInp(ctx, input, other);
}

DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsGtScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsGtScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsMaximum(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsCopyInp(ctx, src, input);
}

DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsLogicalAnd(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsLogicalAnd(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseAnd(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsBitwiseAnd(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseAndScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseAndScalarInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsBitwiseAndScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input,
                                int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsMax(ctx, max, max_indices, input, dim);
}

DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min_val,
                                     const diopiScalar_t *max_val) {
    TOPSOP_LOG();
    return impl::tops::topsHardtanh(ctx, out, input, min_val, max_val);
}

DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min_val, const diopiScalar_t *max_val) {
    TOPSOP_LOG();
    return impl::tops::topsHardtanhInp(ctx, input, min_val, max_val);
}

DIOPI_API diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    TOPSOP_LOG();
    return impl::tops::topsopUpsampleNearest(ctx, out, input, size);
}

DIOPI_API diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                           bool align_corners, const char *mode) {
    TOPSOP_LOG();
    return impl::tops::topsopUpsampleLinear(ctx, out, input, size, align_corners, mode);
}

DIOPI_API diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *p, diopiSize_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsNorm(ctx, out, input, p, dim, diopi_dtype_float32);
}

DIOPI_API diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    TOPSOP_LOG();
    return impl::tops::topsPermute(ctx, out, input, dims);
}

DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsSum(ctx, out, input, dim, diopi_dtype_float32);
}
DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                                 int64_t dim, bool largest, bool sorted) {
    TOPSOP_LOG();
    return impl::tops::topsTopk(ctx, values, indices, input, k, dim, largest, sorted);
}

DIOPI_API diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t *dim, bool keepdim) {
    TOPSOP_LOG();
    return impl::tops::topsArgmax(ctx, out, input, dim, keepdim);
}

DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                                    bool train, diopiGeneratorHandle_t generator) {
    return impl::tops::topsDropout(ctx, out, mask, input, p, train, generator);
}
DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                                       diopiGeneratorHandle_t generator) {
    TOPSOP_LOG();
    return impl::tops::topsDropoutInp(ctx, input, mask, p, train, generator);
}
DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char *mode,
                                const double *value) {
    TOPSOP_LOG();
    return impl::tops::topsPad(ctx, out, input, pad, mode, value);
}

DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                   diopiConstTensorHandle_t bias) {
    TOPSOP_LOG();
    return impl::tops::topsLinear(ctx, out, input, weight, bias);
}
DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                           diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight) {
    TOPSOP_LOG();
    return impl::tops::topsLinearBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight);
}

DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char *approximate) {
    TOPSOP_LOG();
    return impl::tops::topsGelu(ctx, out, input, approximate);
}

DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input, const char *approximate) {
    TOPSOP_LOG();
    return impl::tops::topsGeluBackward(ctx, grad_input, grad_output, input, approximate);
}

DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsMatmul(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes) {
    TOPSOP_LOG();
    return impl::tops::topsOneHot(ctx, out, input, num_classes);
}

DIOPI_API diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    TOPSOP_LOG();
    return impl::tops::topsRoll(ctx, out, input, shifts, dims);
}

DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    TOPSOP_LOG();
    return impl::tops::topsTranspose(ctx, out, input, dim0, dim1);
}

DIOPI_API diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiSize_t normalized_shape, double eps) {
    TOPSOP_LOG();
    return impl::tops::topsLayerNorm(ctx, out, save_mean, save_invstd, input, weight, bias, normalized_shape, eps);
}

DIOPI_API diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean,
                                              diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    TOPSOP_LOG();
    return impl::tops::topsLayerNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias, mean, rstd, normalized_shape);
}

DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t *tensors, int64_t numTensors, int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsStack(ctx, out, tensors, numTensors, dim);
}

DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                                 bool descending, const bool *stable) {
    TOPSOP_LOG();
    return impl::tops::topsSort(ctx, values, indices, input, dim, descending, stable);
}

DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t *min_val, const diopiScalar_t *max_val) {
    TOPSOP_LOG();
    return impl::tops::topsHardtanhBackward(ctx, grad_input, grad_output, input, min_val, max_val);
}

DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsAbs(ctx, out, input);
}

DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t *dim) {
    TOPSOP_LOG();
    return impl::tops::topsAny(ctx, out, input, dim);
}

DIOPI_API diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsAbsInp(ctx, input);
}

DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsLt(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsLtScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsLe(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsLeScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    TOPSOP_LOG();
    return impl::tops::topsClampMin(ctx, out, input, min);
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    TOPSOP_LOG();
    return impl::tops::topsClampMinInp(ctx, input, min);
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min) {
    TOPSOP_LOG();
    return impl::tops::topsClampMinScalar(ctx, out, input, min);
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min) {
    TOPSOP_LOG();
    return impl::tops::topsClampMinInpScalar(ctx, input, min);
}

DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output, int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsSoftmaxBackward(ctx, grad_input, grad_output, output, dim, diopi_dtype_float32);
}

DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    TOPSOP_LOG();
    return impl::tops::topsMaxPool2d(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
}

DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    TOPSOP_LOG();
    return impl::tops::topsMaxPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

DIOPI_API diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *start, const diopiScalar_t *end, int64_t steps) {
    TOPSOP_LOG();
    return impl::tops::topsLinspace(ctx, out, start, end, steps);
}

DIOPI_API diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min, const diopiScalar_t *max) {
    TOPSOP_LOG();
    return impl::tops::topsClampInpScalar(ctx, input, min, max);
}
DIOPI_API diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    TOPSOP_LOG();
    return impl::tops::topsClampInp(ctx, input, min, max);
}
DIOPI_API diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min,
                                        const diopiScalar_t *max) {
    TOPSOP_LOG();
    return impl::tops::topsClampScalar(ctx, out, input, min, max);
}
DIOPI_API diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                                  diopiConstTensorHandle_t max) {
    TOPSOP_LOG();
    return impl::tops::topsClamp(ctx, out, input, min, max);
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *max) {
    TOPSOP_LOG();
    return impl::tops::topsClampMaxInpScalar(ctx, input, max);
}
DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    TOPSOP_LOG();
    return impl::tops::topsClampMaxInp(ctx, input, max);
}
DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *max) {
    TOPSOP_LOG();
    return impl::tops::topsClampMaxScalar(ctx, out, input, max);
}
DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    TOPSOP_LOG();
    return impl::tops::topsClampMax(ctx, out, input, max);
}

DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsLogInp(ctx, input);
}
DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsLog(ctx, out, input);
}

DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsLog2Inp(ctx, input);
}
DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsLog2(ctx, out, input);
}

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsLog10Inp(ctx, input);
}
DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsLog10(ctx, out, input);
}

DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsLogSoftmax(ctx, out, input, dim);
}

DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, int64_t groups) {
    TOPSOP_LOG();
    return impl::tops::topsConvolution2dBackward(
        ctx, grad_input, grad_weight, grad3, grad_output, input, weight, bias_sizes, stride, padding, dilation, groups);
}

DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t *alpha) {
    TOPSOP_LOG();
    return impl::tops::topsAddInp(ctx, input, other, alpha);
}

DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t *alpha) {
    TOPSOP_LOG();
    return impl::tops::topsSubInp(ctx, input, other, alpha);
}

DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsMulInp(ctx, input, other);
}

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsSigmoidInp(ctx, input);
}

DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var,
                                              diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    TOPSOP_LOG();
    return impl::tops::topsBatchNormBackward(
        ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, training, eps);
}

DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsNeScalar(ctx, out, input, other);
}
DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsNe(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsLogSoftmaxBackward(ctx, grad_input, grad_output, output, dim);
}

DIOPI_API diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsExp(ctx, out, input);
}

DIOPI_API diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsExpInp(ctx, input);
}

DIOPI_API diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsMinimum(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsFloor(ctx, out, input);
}

DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsFloorInp(ctx, input);
}

DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    TOPSOP_LOG();
    return impl::tops::topsMean(ctx, out, input, dim, diopi_dtype_float32);
}

DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsReciprocal(ctx, out, input);
}

DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsReciprocalInp(ctx, input);
}

DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseNot(ctx, out, input);
}

DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, diopiGeneratorHandle_t generator) {
    TOPSOP_LOG();
    return impl::tops::topsRandperm(ctx, out, n, generator);
}

DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                             diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    TOPSOP_LOG();
    return impl::tops::topsCrossEntropyLoss(ctx, out, input, target, weight, reduction, ignore_index, label_smoothing);
}

DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    TOPSOP_LOG();
    return impl::tops::topsCrossEntropyLossBackward(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index, label_smoothing);
}

DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum,
                                double dampening, double weight_decay, bool nesterov) {
    TOPSOP_LOG();
    return impl::tops::topsSgd(ctx, w, dw, buf, lr, momentum, dampening, weight_decay, nesterov);
}

DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsExpand(ctx, out, input);
}

DIOPI_API diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                 diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                 float weight_decay, int64_t step, bool amsgrad) {
    TOPSOP_LOG();
    return impl::tops::topsAdam(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    TOPSOP_LOG();
    return impl::tops::topsAdaptiveAvgPool2d(ctx, out, input, output_size);
}

DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsMaxAll(ctx, max, input);
}

DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other, const diopiScalar_t *alpha) {
    TOPSOP_LOG();
    return impl::tops::topsAddInpScalar(ctx, input, other, alpha);
}

DIOPI_API diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                  diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int64_t step, bool amsgrad) {
    TOPSOP_LOG();
    return impl::tops::topsAdamW(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

DIOPI_API diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsNeInp(ctx, input, other);
}

DIOPI_API diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsNeInpScalar(ctx, input, other);
}

DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsLeInp(ctx, input, other);
}

DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsLeInpScalar(ctx, input, other);
}

DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    diopiTensorHandle_t out = input;
    TOPSOP_LOG();
    return impl::tops::topsGeScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsLtInpScalar(ctx, input, other);
}

DIOPI_API diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsLtInp(ctx, input, other);
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsMulInpScalar(ctx, input, other);
}

DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                                      int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    TOPSOP_LOG();
    return impl::tops::topsEmbedding(ctx, out, weight, indices, padding_idx, scale_grad_byfreq, sparse);
}

DIOPI_API diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
    TOPSOP_LOG();
    return impl::tops::topsRepeat(ctx, out, input, repeats_size);
}

DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    TOPSOP_LOG();
    return impl::tops::topsTril(ctx, out, input, diagonal);
}

DIOPI_API diopiError_t diopiTrilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    TOPSOP_LOG();
    return impl::tops::topsTrilInp(ctx, input, diagonal);
}

// DIOPI_API diopiError_t diopiMultinomial(diopiContextHandle_t ctx,
//                                         diopiTensorHandle_t out,
//                                         diopiConstTensorHandle_t input,
//                                         int64_t num_samples,
//                                         bool replacement) {
// TOPSOP_LOG();
//   return impl::tops::topsMultinomial(ctx, out, input, num_samples,
//   replacement);
// }

DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    TOPSOP_LOG();
    return impl::tops::topsBmm(ctx, out, input, mat2);
}

DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                   diopiConstTensorHandle_t index) {
    return impl::tops::topsGather(ctx, out, input, dim, index);
}

DIOPI_API diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    TOPSOP_LOG();
    return impl::tops::topsSelect(ctx, out, input, dim, index);
}

DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    TOPSOP_LOG();
    return impl::tops::topsMm(ctx, out, input, mat2);
}

DIOPI_API diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsRsqrt(ctx, out, input);
}

DIOPI_API diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsRsqrtInp(ctx, input);
}

DIOPI_API diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                        diopiConstTensorHandle_t index) {
    return impl::tops::topsIndexSelect(ctx, out, input, dim, index);
}

DIOPI_API diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    TOPSOP_LOG();
    return impl::tops::topsFlip(ctx, out, input, dims);
}

DIOPI_API diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsCastDtype(ctx, out, input);
}

DIOPI_API diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t *indices,
                                        int64_t indices_counts, bool accumulate) {
    TOPSOP_LOG();
    return impl::tops::topsIndexPutInp(ctx, input, values, indices, indices_counts, accumulate);
}
DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                                     diopiConstTensorHandle_t *indices, int64_t indices_counts, bool accumulate) {
    TOPSOP_LOG();
    return impl::tops::topsIndexPut(ctx, out, input, values, indices, indices_counts, accumulate);
}

// DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx,
//                                    diopiTensorHandle_t out,
//                                    diopiConstTensorHandle_t input,
//                                    int64_t dim) {
//   return impl::tops::topCumsum(ctx, out, input, dim);
// }
DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t *threshold) {
    TOPSOP_LOG()
    return impl::tops::topsThresholdBackward(ctx, grad_input, grad_output, input, threshold);
}

DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                                  diopiConstTensorHandle_t mat2, const diopiScalar_t *beta, const diopiScalar_t *alpha) {
    TOPSOP_LOG();
    return impl::tops::topsAddMm(ctx, out, input, mat1, mat2, beta, alpha);
}

DIOPI_API diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsMinAll(ctx, min, input);
}

DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsNeg(ctx, out, input);
}

DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                          const diopiScalar_t *value, diopiConstTensorHandle_t index, const char *reduce) {
    TOPSOP_LOG();
    return impl::tops::topsScatterScalar(ctx, out, input, dim, value, index, reduce);
}

DIOPI_API diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t *indices,
                                  int64_t nums) {
    TOPSOP_LOG();
    return impl::tops::topsIndex(ctx, out, input, indices, nums);
}

DIOPI_API diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size) {
    TOPSOP_LOG();
    return impl::tops::topsopUpsampleNearestBackward(ctx, grad_input, grad_output, out_size, in_size);
}

DIOPI_API diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                   diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char *mode) {
    TOPSOP_LOG();
    return impl::tops::topsopUpsampleLinearBackward(ctx, grad_input, grad_output, out_size, in_size, align_corners, mode);
}

DIOPI_API diopiError_t diopiContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
    TOPSOP_LOG();
    return impl::tops::topsContiguous(ctx, out, input, memoryFormat);
}

DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t *dim) {
    TOPSOP_LOG();
    return impl::tops::topsProd(ctx, out, input, dim);
}

DIOPI_API diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsSilu(ctx, out, input);
}

DIOPI_API diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsSiluInp(ctx, input);
}

DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsWhere(ctx, out, condition, input, other);
}

DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, const int64_t *dim, bool sorted,
                                   bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t *counts) {
    TOPSOP_LOG();
    return impl::tops::topsUnique(ctx, out, input, dim, sorted, return_counts, indices, counts);
}

DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t *value) {
    TOPSOP_LOG();
    return impl::tops::topsAddcdiv(ctx, out, input, tensor1, tensor2, value);
}

DIOPI_API diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t *value) {
    TOPSOP_LOG();
    return impl::tops::topsAddcdivInp(ctx, input, tensor1, tensor2, value);
}

DIOPI_API diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t *value) {
    TOPSOP_LOG();
    return impl::tops::topsAddcmul(ctx, out, input, tensor1, tensor2, value);
}

DIOPI_API diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t *value) {
    TOPSOP_LOG();
    return impl::tops::topsAddcmulInp(ctx, input, tensor1, tensor2, value);
}

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    return impl::tops::topsMaskedFillInp(ctx, input, mask, value);
}

DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value) {
    return impl::tops::topsMaskedFill(ctx, out, input, mask, value);
}

DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t *value) {
    return impl::tops::topsMaskedFillScalar(ctx, out, input, mask, value);
}

DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask,
                                                const diopiScalar_t *value) {
    return impl::tops::topsMaskedFillInpScalar(ctx, input, mask, value);
}

DIOPI_API diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return impl::tops::topsSqrtInp(ctx, input); }

DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    return impl::tops::topsSqrt(ctx, out, input);
}

DIOPI_API diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    return impl::tops::topsIsNan(ctx, out, input);
}

DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    TOPSOP_LOG();
    return impl::tops::topsNLLLoss(ctx, out, input, target, weight, reduction, ignore_index);
}

DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index) {
    TOPSOP_LOG();
    return impl::tops::topsNLLLossBackward(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index);
}

DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsCos(ctx, out, input);
}

DIOPI_API diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsCosInp(ctx, input);
}

DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsSin(ctx, out, input);
}

DIOPI_API diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsSinInp(ctx, input);
}

DIOPI_API diopiError_t diopiCeil(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsCeil(ctx, out, input);
}

DIOPI_API diopiError_t diopiCeilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    TOPSOP_LOG();
    return impl::tops::topsCeilInp(ctx, input);
}

DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseOr(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseOrScalar(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseOrInp(ctx, input, other);
}

DIOPI_API diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other) {
    TOPSOP_LOG();
    return impl::tops::topsBitwiseOrInpScalar(ctx, input, other);
}

DIOPI_API diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    TOPSOP_LOG();
    return impl::tops::topsTriu(ctx, out, input, diagonal);
}

DIOPI_API diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    TOPSOP_LOG();
    return impl::tops::topsTriuInp(ctx, input, diagonal);
}

}  // namespace topsrider
}  // namespace impl
