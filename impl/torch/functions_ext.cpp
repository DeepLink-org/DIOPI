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

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                  diopiConstTensorHandle_t sin, const bool conj) {
    impl::aten::setCurCtx(ctx);
    auto atX = impl::aten::buildATen(x);
    auto atCos = impl::aten::buildATen(cos);
    auto atSin = impl::aten::buildATen(sin);
    auto atOut = impl::aten::buildATen(out);
    int last_dim = atX.dim() - 1;          // 确定最后一个维度的索引
    auto chunks = atX.chunk(2, last_dim);  // 将 atX 切分为两个部分
    auto x1 = chunks[0];
    auto x2 = chunks[1];
    auto chunks_out = atOut.chunk(2, last_dim);
    auto out1 = chunks_out[0];
    auto out2 = chunks_out[1];
    ext::ops::apply_rotary_cuda(x1, x2, atCos, atSin, out1, out2, conj);
    return diopiSuccess;
}

}  // extern "C"
