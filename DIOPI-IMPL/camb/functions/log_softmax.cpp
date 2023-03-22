#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    std::vector<int> src_input_shape{input_tensor.shape().begin(), input_tensor.shape().end()};
    std::vector<int> src_output_shape{output_tensor.shape().begin(), output_tensor.shape().end()};

    const int input_rank = input_tensor.shape().size();
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

    if (input_tensor.dtype() == diopi_dtype_float64) {
        return diopiDtypeNotSupported;
    }

    cnnlDataType_t out_dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&out_dtype, dtype));
    const void* x_ptr = input_tensor.data();
    void* y_ptr = output_tensor.data();

    const float alpha = 1;
    const float beta = 0;

    CnnlTensorDesc x_desc, y_desc;
    DIOPI_CALL(x_desc.set(input_tensor, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALL(y_desc.set(output_tensor, CNNL_LAYOUT_ARRAY, input_shape));

    DIOPI_CALLCNNL(cnnlSoftmaxForward_v2(handle,
                                         CNNL_SOFTMAX_LOG,
                                         mode_,
                                         CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                                         &alpha,
                                         x_desc.get(),
                                         input_tensor.data(),
                                         &beta,
                                         y_desc.get(),
                                         output_tensor.data()));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx,
                                                diopiTensorHandle_t grad_input,
                                                diopiConstTensorHandle_t grad_output,
                                                diopiConstTensorHandle_t output,
                                                int64_t dim,
                                                diopiDtype_t input_dtype) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_grad = DiopiTensor(grad_input);
    auto output_grad = DiopiTensor(grad_output);
    auto output_tensor = DiopiTensor(output);
    std::vector<int> src_output_shape{output_tensor.shape().begin(), output_tensor.shape().end()};

    const int input_rank = input_grad.shape().size();

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

    CnnlTensorDesc input_grad_desc, output_grad_desc, output_desc;
    DIOPI_CALL(input_grad_desc.set(input_grad, CNNL_LAYOUT_ARRAY, output_shape));
    DIOPI_CALL(output_grad_desc.set(output_grad, CNNL_LAYOUT_ARRAY, output_shape));
    DIOPI_CALL(output_desc.set(output_tensor, CNNL_LAYOUT_ARRAY, output_shape));

    const void* output_ptr = output_tensor.data();
    const void* output_grad_ptr = output_grad.data();
    void* input_grad_ptr = input_grad.data();

    DIOPI_CALLCNNL(cnnlSoftmaxBackward(handle,
                                       CNNL_SOFTMAX_LOG,
                                       mode_,
                                       NULL,
                                       output_desc.get(),
                                       output_ptr,
                                       output_grad_desc.get(),
                                       output_grad_ptr,
                                       NULL,
                                       input_grad_desc.get(),
                                       input_grad_ptr));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
