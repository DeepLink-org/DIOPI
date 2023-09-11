/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/diopirt.h>

#include <algorithm>
#include <iostream>
#include <sstream>

#include "../ascend_tensor.hpp"
#include "acloprunner.hpp"
#include "debug.hpp"
#include "tensor_utils.hpp"

namespace impl {
namespace ascend {

diopiError_t contiguous(diopiContextHandle_t ctx, AscendTensor& src) {
    if (src.isContiguous()) {
        return diopiSuccess;
    }

    // TODO: optimize
    std::vector<int64_t> baseShape = src.getBaseShape();
    diopiSize_t shape = arrayToDiopiSize(const_cast<int64_t*>(src.shape().data()), src.dim());
    diopiSize_t stride = arrayToDiopiSize(const_cast<int64_t*>(src.stride().data()), src.dim());

    diopiTensorHandle_t out = nullptr;
    // NOTE: stride need to be nullptr.
    diopiRequireTensor(ctx, &out, &shape, nullptr, src.dtype(), diopi_device);
    AclOpRunner<4, 1>("AsStrided", ctx)
        .addInput(src.data(), src.getBaseBufferSize(), baseShape, ACL_FORMAT_ND, src.dtype())
        .addConstInput(shape)
        .addConstInput(stride)
        .addConstInput(0, diopi_dtype_int64)
        .addOutput(out)
        .run();

    src = AscendTensor(out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
