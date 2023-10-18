#pragma once

#include <diopi/functions.h>

static int32_t TOPSOP_LOG_LEVEL = 0;

#define PRINT_COLOR_NONE "\033[0m"
#define PRINT_RED "\033[1;31;40m"
#define PRINT_BLUE "\033[1;34;40m"
#define PRINT_GREEN "\033[1;32;40m"
#define PRINT_YELLOW "\033[1;33;40m"

#define TOPSOP_LOG()                                                                                               \
    if (TOPSOP_LOG_LEVEL) {                                                                                        \
        fprintf(stdout, PRINT_BLUE " TOPSOP_LOG> [Function: %s @ Line: %d] " PRINT_GREEN, __FUNCTION__, __LINE__); \
        fprintf(stdout, PRINT_COLOR_NONE "\n");                                                                    \
    }

namespace impl::tops {

DIOPI_API diopiError_t topsArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *start, const diopiScalar_t *end,
                                  const diopiScalar_t *step);
DIOPI_API diopiError_t topsAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                               const diopiScalar_t *alpha_value);
DIOPI_API diopiError_t topsAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                     const diopiScalar_t * /*alpha*/);
DIOPI_API diopiError_t topsSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                               const diopiScalar_t * /*alpha_value*/);
DIOPI_API diopiError_t topsSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                     const diopiScalar_t * /*alpha*/);

DIOPI_API diopiError_t topsMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                               diopiRoundMode_t /*rounding_mode*/);

DIOPI_API diopiError_t topsDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other,
                                     diopiRoundMode_t /*rounding_mode*/);

DIOPI_API diopiError_t topsPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent);

DIOPI_API diopiError_t topsPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *input, diopiConstTensorHandle_t exponent);

DIOPI_API diopiError_t topsPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *exponent);

DIOPI_API diopiError_t topsConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                         diopiConstTensorHandle_t bias, diopiSize_t stride_, diopiSize_t padding_, diopiSize_t dilation_, int64_t groups);

diopiError_t topsSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t /*dtype*/);

DIOPI_API diopiError_t topsSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsSigmoidInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight, diopiReduction_t reduction);

DIOPI_API diopiError_t topsBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction);

DIOPI_API diopiError_t topsBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                     diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, bool /*training*/, double /*momentum*/,
                                     double eps);

DIOPI_API diopiError_t topsBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                         diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);
DIOPI_API diopiError_t topsSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype);

DIOPI_API diopiError_t topsFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *value);
DIOPI_API diopiError_t topsCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t *tensors, int64_t num_inputs, int64_t dim);

DIOPI_API diopiError_t topsRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *negative_slope);

DIOPI_API diopiError_t topsLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *negative_slope);

DIOPI_API diopiError_t topsLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t *negative_slope, bool input_is_result);

DIOPI_API diopiError_t topsMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiReduction_t reduction);

DIOPI_API diopiError_t topsMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction);

DIOPI_API diopiError_t topsEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsDivScalarInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other, diopiRoundMode_t /*rounding_mode*/);

DIOPI_API diopiError_t topsDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t /*rounding_mode*/);

DIOPI_API diopiError_t topsBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim);

DIOPI_API diopiError_t topsHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min_val,
                                    const diopiScalar_t *max_val);

DIOPI_API diopiError_t topsHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min_val, const diopiScalar_t *max_val);

DIOPI_API diopiError_t topsopUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size);

DIOPI_API diopiError_t topsopUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                            bool align_corners, const char *mode);

DIOPI_API diopiError_t topsNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *p, diopiSize_t dim,
                                diopiDtype_t dtype);

DIOPI_API diopiError_t topsPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

DIOPI_API diopiError_t topsTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                                int64_t dim, bool largest, bool sorted);

DIOPI_API diopiError_t topsArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t *dim, bool keepdim);

DIOPI_API diopiError_t topsPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char *mode,
                               const double *value);

DIOPI_API diopiError_t topsDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                                   bool train, diopiGeneratorHandle_t generator);

DIOPI_API diopiError_t topsDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                                      diopiGeneratorHandle_t generator);

DIOPI_API diopiError_t topsLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias);

DIOPI_API diopiError_t topsLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                          diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight);

DIOPI_API diopiError_t topsGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char *approximate);

DIOPI_API diopiError_t topsGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                        diopiConstTensorHandle_t input, const char *approximate);

DIOPI_API diopiError_t topsMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

// DIOPI_API diopiError_t topsNonzero(diopiContextHandle_t ctx,
//                                    diopiTensorHandle_t* out,
//                                    diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes);

DIOPI_API diopiError_t topsRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims);

DIOPI_API diopiError_t topsTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1);

DIOPI_API diopiError_t topsLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                     diopiSize_t normalized_shape, double eps);

DIOPI_API diopiError_t topsLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                             diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                             diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean,
                                             diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape);

DIOPI_API diopiError_t topsStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t *tensors, int64_t numTensors, int64_t dim);

DIOPI_API diopiError_t topsSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                                bool descending, const bool *stable);

DIOPI_API diopiError_t topsHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min_val,
                                    const diopiScalar_t *max_val);

DIOPI_API diopiError_t topsHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min_val, const diopiScalar_t *max_val);

DIOPI_API diopiError_t topsHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, const diopiScalar_t *min_val, const diopiScalar_t *max_val);

DIOPI_API diopiError_t topsAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t *dim);

DIOPI_API diopiError_t topsAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min);

DIOPI_API diopiError_t topsClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min);

DIOPI_API diopiError_t topsClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min);

DIOPI_API diopiError_t topsClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min);

DIOPI_API diopiError_t topsSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype);

DIOPI_API diopiError_t topsMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                     diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

DIOPI_API diopiError_t topsMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                             diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);
DIOPI_API diopiError_t topsLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *start, const diopiScalar_t *end, int64_t steps);

DIOPI_API diopiError_t topsClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *min, const diopiScalar_t *max);

DIOPI_API diopiError_t topsClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max);

DIOPI_API diopiError_t topsClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *min,
                                       const diopiScalar_t *max);

DIOPI_API diopiError_t topsClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                                 diopiConstTensorHandle_t max);

DIOPI_API diopiError_t topsClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *max);
DIOPI_API diopiError_t topsClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max);
DIOPI_API diopiError_t topsClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *max);
DIOPI_API diopiError_t topsClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max);

DIOPI_API diopiError_t topsLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t topsLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t topsLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t topsLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);
DIOPI_API diopiError_t topsLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t output, int64_t dim);

DIOPI_API diopiError_t topsConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                 diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                 diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                 diopiSize_t dilation, int64_t groups);

DIOPI_API diopiError_t topsAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t *alpha);

DIOPI_API diopiError_t topsSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t *alpha);

DIOPI_API diopiError_t topsMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                             diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                             diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var,
                                             diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps);

DIOPI_API diopiError_t topsNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t topsExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype);

DIOPI_API diopiError_t topsReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsReciprocalInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                            diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);

DIOPI_API diopiError_t topsCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                    diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);

DIOPI_API diopiError_t topsSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum,
                               double dampening, double weight_decay, bool nesterov);

DIOPI_API diopiError_t topsExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                float weight_decay, int64_t step, bool amsgrad);

DIOPI_API diopiError_t topsAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);

DIOPI_API diopiError_t topsNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, diopiGeneratorHandle_t generator);

DIOPI_API diopiError_t topsMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other, const diopiScalar_t *alpha);

DIOPI_API diopiError_t topsAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                 diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                 float weight_decay, int64_t step, bool amsgrad);

DIOPI_API diopiError_t topsNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                                     int64_t padding_idx, bool scale_grad_byfreq, bool sparse);

DIOPI_API diopiError_t topsEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                             int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse);

DIOPI_API diopiError_t topsRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size);

DIOPI_API diopiError_t topsTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal);
DIOPI_API diopiError_t topsTrilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal);

DIOPI_API diopiError_t topsMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples,
                                       bool replacement);

DIOPI_API diopiError_t topsBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

DIOPI_API diopiError_t topsGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                  diopiConstTensorHandle_t index);

DIOPI_API diopiError_t topsSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index);
DIOPI_API diopiError_t topsMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

DIOPI_API diopiError_t topsRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                       diopiConstTensorHandle_t index);
DIOPI_API diopiError_t topsFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

DIOPI_API diopiError_t topsCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t *threshold);

DIOPI_API diopiError_t topsNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std);

DIOPI_API diopiError_t topsIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t *indices,
                                       int64_t indices_counts, bool accumulate);

DIOPI_API diopiError_t topsIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                                    diopiConstTensorHandle_t *indices, int64_t indices_counts, bool accumulate);

DIOPI_API diopiError_t topCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);

DIOPI_API diopiError_t topsAddMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                                 diopiConstTensorHandle_t mat2, const diopiScalar_t *beta, const diopiScalar_t *alpha);

DIOPI_API diopiError_t topsMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsNms(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores,
                               double iou_threshold, int64_t offset);

DIOPI_API diopiError_t topsScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                         const diopiScalar_t *value, diopiConstTensorHandle_t index, const char *reduce);

DIOPI_API diopiError_t topsIndex(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t *indices,
                                 int64_t nums);

DIOPI_API diopiError_t topsopUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiSize_t out_size, diopiSize_t in_size);

DIOPI_API diopiError_t topsopUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char *mode);

DIOPI_API diopiError_t topsContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat);

DIOPI_API diopiError_t topsUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                                  bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts);

DIOPI_API diopiError_t topsProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t *dim);

DIOPI_API diopiError_t topsSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t topsSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                                 diopiConstTensorHandle_t other);

DIOPI_API diopiError_t topsAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                   diopiConstTensorHandle_t tensor2, const diopiScalar_t *value);

DIOPI_API diopiError_t topsAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                      const diopiScalar_t *value);

DIOPI_API diopiError_t topsAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                   diopiConstTensorHandle_t tensor2, const diopiScalar_t *value);

DIOPI_API diopiError_t topsAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                      const diopiScalar_t *value);

DIOPI_API diopiError_t topsMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                      diopiConstTensorHandle_t value);

DIOPI_API diopiError_t topsMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value);

DIOPI_API diopiError_t topsMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                            const diopiScalar_t *value);

DIOPI_API diopiError_t topsMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t *value);

DIOPI_API diopiError_t topsSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t topsNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index);

DIOPI_API diopiError_t topsNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction, int64_t ignore_index);

DIOPI_API diopiError_t topsCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t topsSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t topsCeil(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t topsCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t topsSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t topsCeilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

DIOPI_API diopiError_t topsBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t topsBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other);
DIOPI_API diopiError_t topsBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t topsBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t *other);

DIOPI_API diopiError_t topsTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal);
DIOPI_API diopiError_t topsTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal);

int topsLibInit();

int topsLibFinalize();

}  // namespace impl::tops
