/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include <iostream>

#include "context.h"
#include "ext_kernel.h"
#include "helper.hpp"

extern "C" {

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out1, diopiTensorHandle_t out2, diopiConstTensorHandle_t x1,
                                  diopiConstTensorHandle_t x2, diopiConstTensorHandle_t cos, diopiConstTensorHandle_t sin, const bool conj) {
    impl::aten::setCurCtx(ctx);
    auto atOut1 = impl::aten::buildATen(out1);
    auto atOut2 = impl::aten::buildATen(out2);
    auto atX1 = impl::aten::buildATen(x1);
    auto atX2 = impl::aten::buildATen(x2);
    auto atCos = impl::aten::buildATen(cos);
    auto atSin = impl::aten::buildATen(sin);
    ext::ops::apply_rotary_cuda(atX1, atX2, atCos, atSin, atOut1, atOut2, conj);
    return diopiSuccess;
}

}  // extern "C"
