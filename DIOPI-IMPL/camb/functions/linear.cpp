/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t flatten_to_2d(std::vector<long int> in_dims, std::vector<int>& out_dims, bool keep_first = true) {
    out_dims.resize(2);
    if (in_dims.size() == 2) {
        out_dims[0] = in_dims[0];
        out_dims[1] = in_dims[1];
    } else if (in_dims.size() > 2) {
        if (keep_first) {
            out_dims[0] = in_dims[0];
            out_dims[1] = std::accumulate(in_dims.begin() + 1, in_dims.end(), 1, std::multiplies<int32_t>());
        } else {
            out_dims[0] = std::accumulate(in_dims.begin(), in_dims.end() - 1, 1, std::multiplies<int32_t>());
            out_dims[1] = in_dims[in_dims.size() - 1];
        }
    } else {
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}
}  // namespace

extern "C" diopiError_t diopiLinear(
    diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);
    auto weight_tensor = DiopiTensor(weight);

    std::vector<int> input_shape, weight_shape, output_shape;
    DIOPI_CALL(flatten_to_2d(input_tensor.shape(), input_shape, false));
    DIOPI_CALL(flatten_to_2d(weight_tensor.shape(), weight_shape, false));
    DIOPI_CALL(flatten_to_2d(output_tensor.shape(), output_shape, false));

    CnnlTensorDesc input_desc, weight_desc, output_desc;
    DIOPI_CALL(input_desc.set(input_tensor, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALL(weight_desc.set(weight_tensor, CNNL_LAYOUT_ARRAY, weight_shape));
    DIOPI_CALL(output_desc.set(output_tensor, CNNL_LAYOUT_ARRAY, output_shape));

    float alpha = 1.0;
    float beta = 0.0;

    if (bias != nullptr) {
        beta = 1.0;
        auto bias_tensor = DiopiTensor(bias);
        CnnlTensorDesc bias_desc(bias_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlExpand(handle, bias_desc.get(), bias_tensor.data(), output_desc.get(), output_tensor.data()));
    }

    DIOPI_CALLCNNL(cnnlMatMul(handle,
                              false,
                              true,
                              &alpha,
                              input_desc.get(),
                              input_tensor.data(),
                              weight_desc.get(),
                              weight_tensor.data(),
                              &beta,
                              output_desc.get(),
                              output_tensor.data()));
    return diopiSuccess;
}
extern "C" diopiError_t diopiLinearBackward(diopiContextHandle_t ctx,
                                            diopiTensorHandle_t grad_input,
                                            diopiTensorHandle_t grad_weight,
                                            diopiTensorHandle_t grad_bias,
                                            diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_weight_tensor = DiopiTensor(grad_weight);
    auto grad_output_tensor = DiopiTensor(grad_output);
    auto input_tensor = DiopiTensor(input);
    auto weight_tensor = DiopiTensor(weight);

    std::vector<int> input_shape, weight_shape, output_shape;
    DIOPI_CALL(flatten_to_2d(input_tensor.shape(), input_shape, false));
    DIOPI_CALL(flatten_to_2d(weight_tensor.shape(), weight_shape, false));
    DIOPI_CALL(flatten_to_2d(grad_output_tensor.shape(), output_shape, false));

    CnnlTensorDesc grad_input_desc, grad_weight_desc, grad_output_desc, input_desc, weight_desc;
    DIOPI_CALL(grad_input_desc.set(grad_input_tensor, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALL(grad_weight_desc.set(grad_weight_tensor, CNNL_LAYOUT_ARRAY, weight_shape));
    DIOPI_CALL(grad_output_desc.set(grad_output_tensor, CNNL_LAYOUT_ARRAY, output_shape));
    DIOPI_CALL(input_desc.set(input_tensor, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALL(weight_desc.set(weight_tensor, CNNL_LAYOUT_ARRAY, weight_shape));

    float alpha = 1.0;
    float beta = 0.0;

    DIOPI_CALLCNNL(cnnlMatMul(handle,
                              true,
                              false,
                              &alpha,
                              grad_output_desc.get(),
                              grad_output_tensor.data(),
                              input_desc.get(),
                              input_tensor.data(),
                              &beta,
                              grad_weight_desc.get(),
                              grad_weight_tensor.data()));

    DIOPI_CALLCNNL(cnnlMatMul(handle,
                              false,
                              false,
                              &alpha,
                              grad_output_desc.get(),
                              grad_output_tensor.data(),
                              weight_desc.get(),
                              weight_tensor.data(),
                              &beta,
                              grad_input_desc.get(),
                              grad_input_tensor.data()));

    if (grad_bias != nullptr) {
        auto bias_grad_tensor = DiopiTensor(grad_bias);
        CnnlTensorDesc bias_grad_desc;
        DIOPI_CALL(bias_grad_desc.set(bias_grad_tensor, CNNL_LAYOUT_ARRAY));
        std::vector<int64_t> bias_shape{bias_grad_tensor.shape().begin(), bias_grad_tensor.shape().end()};
        size_t workspace_size_bias;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, grad_output_desc.get(), bias_grad_desc.get(), 3, &workspace_size_bias))

        void* workspace_bias = nullptr;
        if (0 != workspace_size_bias) {
            workspace_bias = requiresBuffer(ctx, workspace_size_bias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(
            handle, grad_output_desc.get(), grad_output_tensor.data(), 1, bias_grad_desc.get(), bias_grad_tensor.data(), workspace_bias, workspace_size_bias));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
