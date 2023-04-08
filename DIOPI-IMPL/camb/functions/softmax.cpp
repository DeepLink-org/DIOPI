/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t softmax_forward(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor output, int64_t dim, bool is_log = false) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_casted = input;
    DiopiTensor output_casted = output;

    std::vector<DiopiTensor*> tensors{&input_casted, &output_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    std::vector<int> src_input_shape{input_casted.shape().begin(), input_casted.shape().end()};
    std::vector<int> src_output_shape{output_casted.shape().begin(), output_casted.shape().end()};

    const int input_rank = input_casted.shape().size();
    int mode = dim;
    mode = (mode < 0) ? (mode + input_rank) : mode;
    const size_t input_dim = 3;
    std::vector<int> input_shape(input_dim, 1);
    if (input_rank != 0) {
        if (input_rank <= 3) {
            input_shape[2] = src_input_shape[input_rank - 1];
            input_shape[1] = (input_rank == 1) ? 1 : src_input_shape[input_rank - 2];
            input_shape[0] = (input_rank == 3) ? src_input_shape[0] : 1;
        } else {
            auto reduce_dim = [](const std::vector<int>& data, int from, int to) -> int {
                to = std::min<int>(to, data.size());
                from = std::max<int>(0, from);
                return std::accumulate(data.cbegin() + from, data.cbegin() + to + 1, 1LL, std::multiplies<int64_t>());
            };
            const bool flag = (mode == input_rank - 1);
            input_shape[0] = reduce_dim(src_input_shape, 0, flag ? (mode - 2) : (mode - 1));
            input_shape[1] = src_input_shape[flag ? (mode - 1) : mode];
            input_shape[2] = reduce_dim(src_input_shape, flag ? mode : (mode + 1), (input_rank - 1));
        }
    }
    cnnlSoftmaxMode_t mode_;
    if (input_rank == 3 && mode == 0) {
        mode_ = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
    } else if (mode == input_rank - 1) {
        mode_ = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    } else {
        mode_ = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    }

    const float alpha = 1;
    const float beta = 0;

    CnnlTensorDesc x_desc, y_desc;
    DIOPI_CALL(x_desc.set(input_casted, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALL(y_desc.set(output_casted, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALLCNNL(cnnlSoftmaxForward_v2(handle,
                                         is_log ? CNNL_SOFTMAX_LOG : CNNL_SOFTMAX_ACCURATE,
                                         mode_,
                                         CNNL_COMPUTATION_FAST,
                                         &alpha,
                                         x_desc.get(),
                                         input_casted.data(),
                                         &beta,
                                         y_desc.get(),
                                         output_casted.data()));

    DIOPI_CALL(dataTypeCast(ctx, output, output_casted));
    return diopiSuccess;
}

diopiError_t softmax_backward(diopiContextHandle_t ctx, DiopiTensor grad_input_tensor, DiopiTensor grad_output_tensor, DiopiTensor output_tensor, int64_t dim,
                              bool is_log = false) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor grad_input_casted = grad_input_tensor;
    DiopiTensor grad_output_casted = grad_output_tensor;
    DiopiTensor output_casted = output_tensor;

    std::vector<DiopiTensor*> tensors{&grad_input_casted, &grad_output_casted, &output_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    std::vector<int> src_output_shape{output_casted.shape().begin(), output_casted.shape().end()};

    const int input_rank = grad_input_casted.shape().size();

    const size_t input_dim = 3;
    int mode = dim;
    std::vector<int> output_shape(input_dim, 1);
    if (input_rank != 0) {
        if (input_rank <= 3) {
            output_shape[2] = src_output_shape[input_rank - 1];
            output_shape[1] = (input_rank == 1) ? 1 : src_output_shape[input_rank - 2];
            output_shape[0] = (input_rank == 3) ? src_output_shape[0] : 1;
        } else {
            auto reduce_dim = [](const std::vector<int>& data, int from, int to) -> int {
                to = std::min<int>(to, data.size());
                from = std::max<int>(0, from);
                return std::accumulate(data.cbegin() + from, data.cbegin() + to + 1, 1LL, std::multiplies<int64_t>());
            };
            const bool flag = (mode == input_rank - 1);
            output_shape[0] = reduce_dim(src_output_shape, 0, flag ? (mode - 2) : (mode - 1));
            output_shape[1] = src_output_shape[flag ? (mode - 1) : mode];
            output_shape[2] = reduce_dim(src_output_shape, flag ? mode : (mode + 1), (input_rank - 1));
        }
    }

    mode = (mode < 0) ? (mode + input_rank) : mode;

    cnnlSoftmaxMode_t mode_;
    if (input_rank == 3 && mode == 0) {
        mode_ = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
    } else if (mode == input_rank - 1) {
        mode_ = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    } else {
        mode_ = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    }

    CnnlTensorDesc grad_input_desc, grad_output_desc, output_desc;
    DIOPI_CALL(grad_input_desc.set(grad_input_casted, CNNL_LAYOUT_ARRAY, output_shape));
    DIOPI_CALL(grad_output_desc.set(grad_output_casted, CNNL_LAYOUT_ARRAY, output_shape));
    DIOPI_CALL(output_desc.set(output_casted, CNNL_LAYOUT_ARRAY, output_shape));

    DIOPI_CALLCNNL(cnnlSoftmaxBackward(handle,
                                       is_log ? CNNL_SOFTMAX_LOG : CNNL_SOFTMAX_ACCURATE,
                                       mode_,
                                       NULL,
                                       output_desc.get(),
                                       output_casted.data(),
                                       grad_output_desc.get(),
                                       grad_output_casted.data(),
                                       NULL,
                                       grad_input_desc.get(),
                                       grad_input_casted.data()));
    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, grad_input_casted));
    return diopiSuccess;
}

}  // namespace

extern "C" diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(softmax_forward(ctx, input_tensor, output_tensor, dim));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t output, int64_t dim) {
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor output_tensor(output);
    DIOPI_CALL(softmax_backward(ctx, grad_input_tensor, grad_output_tensor, output_tensor, dim));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(softmax_forward(ctx, input_tensor, output_tensor, dim, true));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                diopiConstTensorHandle_t output, int64_t dim) {
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor output_tensor(output);
    DIOPI_CALL(softmax_backward(ctx, grad_input_tensor, grad_output_tensor, output_tensor, dim, true));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
