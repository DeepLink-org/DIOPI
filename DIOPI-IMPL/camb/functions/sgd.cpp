/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t w,
                                           diopiTensorHandle_t dw,
                                           diopiTensorHandle_t buf,
                                           double lr,
                                           double momentum,
                                           double dampening,
                                           double weight_decay,
                                           bool nesterov) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto w_tensor = makeTensor(w);
    auto dw_tensor = makeTensor(dw);
    auto buf_tensor = makeTensor(buf);

    CnnlTensorDesc w_desc(w_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc dw_desc(dw_tensor, CNNL_LAYOUT_ARRAY);

    // a = a * scale_a + b * scale_b;
    auto add_mul_func = [&](auto &a, float scale_a, auto b, float scale_b) {
        size_t workspace_size;
        std::vector<int> shape;
        shape.push_back(a.numel());
        CnnlTensorDesc a_desc, b_desc;
        DIOPI_CALL(a_desc.set(a, CNNL_LAYOUT_ARRAY, shape));
        DIOPI_CALL(b_desc.set(b, CNNL_LAYOUT_ARRAY, shape));

        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, b_desc.get(), a_desc.get(), &workspace_size));

        void *workspace = nullptr;
        if (workspace_size != 0) {
            workspace = requiresBuffer(ctx, workspace_size).data();
        }

        DIOPI_CALLCNNL(cnnlBiasAdd(handle, &scale_b, b_desc.get(), b.data(), workspace, workspace_size, &scale_a, a_desc.get(), a.data()));
        return diopiSuccess;
    };

    if (weight_decay != 0) {
        DIOPI_CALL(add_mul_func(dw_tensor, 1.0, w_tensor, weight_decay));
    }
    if (momentum != 0) {
        if (buf == nullptr) {
            if (nesterov) {
                DIOPI_CALL(add_mul_func(dw_tensor, 1.0, dw_tensor, momentum));
            }
        } else {
            auto buf_tensor = makeTensor(buf);
            CnnlTensorDesc buf_desc(buf_tensor, CNNL_LAYOUT_ARRAY);
            DIOPI_CALL(add_mul_func(buf_tensor, momentum, dw_tensor, (1.0 - dampening)));
            if (nesterov) {
                DIOPI_CALL(add_mul_func(dw_tensor, 1.0, buf_tensor, momentum));
            } else {
                DIOPI_CALLCNNL(cnnlCopy(handle, buf_desc.get(), buf_tensor.data(), dw_desc.get(), dw_tensor.data()));
            }
        }
    }

    std::vector<int64_t> shape{1};
    diopiSize_t size(shape.data(), shape.size());
    auto lr_tensor = requiresTensor(ctx, size, diopi_dtype_float32);

    float learning_rate = lr;
    CnnlTensorDesc lr_desc;
    DIOPI_CALL(lr_desc.set(lr_tensor, CNNL_LAYOUT_ARRAY, {1}));
    DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &learning_rate, lr_desc.get(), lr_tensor.data()));
    if (dw_tensor.dtype() == diopi_dtype_float16) {
        CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> lr_half_desc;
        std::vector<int> shape{1};
        DIOPI_CALLCNNL(cnnlSetTensorDescriptor(lr_half_desc.get(), CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 1, shape.data()));
        cnnlCastDataType(handle, lr_desc.get(), lr_tensor.data(), CNNL_CAST_FLOAT_TO_HALF, lr_half_desc.get(), lr_tensor.data());
    }
    DIOPI_CALLCNNL(cnnlGradientDescent(handle, dw_desc.get(), dw_tensor.data(), lr_tensor.data(), w_desc.get(), w_tensor.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
