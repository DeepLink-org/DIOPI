#include <diopi/functions.h>

#include <numeric>
#include <vector>
#include "../common/acloprunner.hpp"

namespace impl {
namespace camb {
extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t out,
                                           diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias,
                                           diopiSize_t stride,
                                           diopiSize_t padding,
                                           diopiSize_t dilation,
                                           int64_t groups) {

    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx,
                                                   diopiTensorHandle_t grad_input,
                                                   diopiTensorHandle_t grad_weight,
                                                   diopiTensorHandle_t grad3,
                                                   diopiConstTensorHandle_t grad_output,
                                                   diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight,
                                                   diopiSize_t *bias_sizes,
                                                   diopiSize_t stride,
                                                   diopiSize_t padding,
                                                   diopiSize_t dilation,
                                                   bool transposed,
                                                   diopiSize_t output_padding,
                                                   int64_t groups) {

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
