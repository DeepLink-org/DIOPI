/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {

std::vector<int> getDim(const DiopiTensor& tensor) {
    int shape_size = tensor.shape().size();
    std::vector<int> dim;
    for (int i = 0; i < shape_size; i++) {
        dim.push_back(static_cast<int>(tensor.shape()[i]));
    }
    if (shape_size == 3) {
        dim.insert(dim.begin(), 1);
    }
    return dim;
}

}  // namespace

extern "C" {
diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    DIOPI_CHECK(input_tensor.dim() == 3 || input_tensor.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&input_tensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    std::vector<int> input_dim = getDim(input_tensor_tmp);
    std::vector<int> out_dim = getDim(out_tensor_tmp);
    CnnlTensorDesc input_desc;
    CnnlTensorDesc out_desc;
    input_desc.set(input_tensor_tmp, CNNL_LAYOUT_NCHW, input_dim);
    out_desc.set(out_tensor_tmp, CNNL_LAYOUT_NCHW, out_dim);

    const int64_t kernel_h = kernel_size.data[0];
    const int64_t kernel_w = kernel_size.len == 1 ? kernel_h : kernel_size.data[1];
    int64_t stride_h = 0;
    int64_t stride_w = 0;
    if (stride.len == 0) {
        stride_h = kernel_h;
        stride_w = kernel_w;
    } else {
        stride_h = stride.data[0];
        stride_w = stride.len == 1 ? stride_h : stride.data[1];
    }
    const int64_t pad_h = padding.data[0];
    const int64_t pad_w = padding.len == 1 ? pad_h : padding.data[1];
    const int64_t dilation_0 = dilation.data[0];
    const int64_t dilation_1 = dilation.len == 1 ? dilation_0 : dilation.data[1];

    // calculate padding coefficients
    auto pl = 0, pr = 0, pu = 0, pd = 0;
    pu = pd = pad_h;
    pl = pr = pad_w;
    if (ceil_mode) {
        // diff = (out - 1) * stride + kernel_size - input
        int diff_height = (out_dim[2] - 1) * stride_h + kernel_h - input_dim[2];
        int diff_width = (out_dim[3] - 1) * stride_w + kernel_w - input_dim[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diff_height > pad_h ? diff_height - pad_h : 0;
        pr = diff_width > pad_w ? diff_width - pad_w : 0;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> CnnlPoolDesc;
    cnnlPoolingDescriptor_t pool_desc = CnnlPoolDesc.get();
    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        pool_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernel_h, kernel_w, pu, pd, pl, pr, stride_h, stride_w, dilation_0, dilation_1, ceil_mode));

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWorkspaceSize(handle, CNNL_POOLING_MAX, out_tensor.shape()[3], input_tensor.shape()[2], &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    const void* alpha = nullptr;
    const void* beta = nullptr;
    DIOPI_CALLCNNL(cnnlPoolingForward(
        handle, pool_desc, alpha, input_desc.get(), input_tensor_tmp.data(), beta, out_desc.get(), out_tensor_tmp.data(), workspace, workspace_size));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));

    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    DiopiTensor indices_tensor(indices);

    DIOPI_CHECK(input_tensor.dim() == 3 || input_tensor.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&input_tensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    DiopiTensor indices_tensor_tmp = indices_tensor;
    if (indices_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int32));
    }
    if (input_tensor_tmp.dtype() == diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int16));
    } else if (input_tensor_tmp.dtype() == diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int32));
    } else {
        DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
    }

    std::vector<int> input_dim = getDim(input_tensor_tmp);
    std::vector<int> indices_dim = getDim(indices_tensor_tmp);
    std::vector<int> out_dim = getDim(out_tensor_tmp);
    CnnlTensorDesc input_desc;
    CnnlTensorDesc indices_desc;
    CnnlTensorDesc out_desc;
    input_desc.set(input_tensor_tmp, CNNL_LAYOUT_NCHW, input_dim);
    indices_desc.set(indices_tensor_tmp, CNNL_LAYOUT_NCHW, indices_dim);
    out_desc.set(out_tensor_tmp, CNNL_LAYOUT_NCHW, out_dim);

    const int64_t kernel_h = kernel_size.data[0];
    const int64_t kernel_w = kernel_size.len == 1 ? kernel_h : kernel_size.data[1];
    int64_t stride_h = 0;
    int64_t stride_w = 0;
    if (stride.len == 0) {
        stride_h = kernel_h;
        stride_w = kernel_w;
    } else {
        stride_h = stride.data[0];
        stride_w = stride.len == 1 ? stride_h : stride.data[1];
    }
    const int64_t pad_h = padding.data[0];
    const int64_t pad_w = padding.len == 1 ? pad_h : padding.data[1];
    const int64_t dilation_0 = dilation.data[0];
    const int64_t dilation_1 = dilation.len == 1 ? dilation_0 : dilation.data[1];

    // calculate padding coefficients
    auto pl = 0, pr = 0, pu = 0, pd = 0;
    pu = pd = pad_h;
    pl = pr = pad_w;
    if (ceil_mode) {
        // diff = (out - 1) * stride + kernel_size - input
        int diff_height = (out_dim[2] - 1) * stride_h + kernel_h - input_dim[2];
        int diff_width = (out_dim[3] - 1) * stride_w + kernel_w - input_dim[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diff_height > pad_h ? diff_height - pad_h : 0;
        pr = diff_width > pad_w ? diff_width - pad_w : 0;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> CnnlPoolDesc;
    cnnlPoolingDescriptor_t pool_desc = CnnlPoolDesc.get();
    int pool_rank_ = kernel_size.len;
    if (pool_rank_ == 3) {
        std::vector<int> window_{kernel_size.data, kernel_size.data + kernel_size.len};
        std::vector<int> padding_{padding.data, padding.data + padding.len};
        std::vector<int> stride_{stride.data, stride.data + stride.len};
        std::vector<int> dilation_{dilation.data, dilation.data + dilation.len};
        DIOPI_CALLCNNL(cnnlSetPoolingNdDescriptor_v2(
            pool_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, pool_rank_ + 2, window_.data(), padding_.data(), stride_.data(), dilation_.data(), ceil_mode));
    } else {
        DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
            pool_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernel_h, kernel_w, pu, pd, pl, pr, stride_h, stride_w, dilation_0, dilation_1, ceil_mode));
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWithIndexWorkspaceSize(handle, input_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(handle,
                                               pool_desc,
                                               nullptr,
                                               input_desc.get(),
                                               input_tensor_tmp.data(),
                                               nullptr,
                                               out_desc.get(),
                                               out_tensor_tmp.data(),
                                               indices_desc.get(),
                                               indices_tensor_tmp.data(),
                                               workspace,
                                               workspace_size));

    if (indices_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int32));
    }
    DIOPI_CALL(dataTypeCast(ctx, indices_tensor, indices_tensor_tmp));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));

    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor indices_tensor(indices);

    DIOPI_CHECK(input_tensor.dim() == 3 || input_tensor.dim() == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    std::vector<DiopiTensor*> pTensors{&input_tensor, &grad_output_tensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor grad_output_tensor_tmp = *pTensors[1];
    DiopiTensor grad_input_tensor_tmp = grad_input_tensor;
    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor_tmp, input_tensor_tmp.dtype()));

    DiopiTensor indices_tensor_tmp = indices_tensor;
    if (indices_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int32));
    }
    if (input_tensor_tmp.dtype() == diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int16));
    } else if (input_tensor_tmp.dtype() == diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor_tmp, diopi_dtype_int32));
    } else {
        DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
    }

    diopiTensorHandle_t input_t = nullptr;
    diopiTensorHandle_t grad_input_t = nullptr;
    diopiTensorHandle_t grad_output_t = nullptr;
    diopiTensorHandle_t indices_t = nullptr;

    auto permute_to_nhwc = [&](auto src, auto& dst) {
        DiopiTensor src_tensor(src);
        std::vector<int64_t> src_shape_t_64(src_tensor.shape().size());
        std::vector<int64_t> axis{0, 2, 3, 1};
        if (src_tensor.shape().size() == 3) {
            axis.clear();
            axis.push_back(1);
            axis.push_back(2);
            axis.push_back(0);
        }
        for (int i = 0; i < src_tensor.shape().size(); ++i) {
            src_shape_t_64[i] = src_tensor.shape()[axis[i]];
        }

        diopiSize_t src_t_shape(src_shape_t_64.data(), src_shape_t_64.size());
        DIOPI_CALL(diopiRequireTensor(ctx, &dst, &src_t_shape, nullptr, src_tensor.dtype(), diopi_device));
        if (src_tensor.shape().size() == 4) {
            diopiSize_t nchw2nhwc(axis.data(), 4);
            DIOPI_CALL(diopiPermute(ctx, dst, src, nchw2nhwc));
        } else if (src_tensor.shape().size() == 3) {
            diopiSize_t chw2hwc(axis.data(), 3);
            DIOPI_CALL(diopiPermute(ctx, dst, src, chw2hwc));
        } else {
            DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
        }
        return diopiSuccess;
    };

    DIOPI_CALL(permute_to_nhwc(static_cast<diopiTensorHandle_t>(input_tensor_tmp), input_t));
    DIOPI_CALL(permute_to_nhwc(static_cast<diopiTensorHandle_t>(grad_input_tensor_tmp), grad_input_t));
    DIOPI_CALL(permute_to_nhwc(static_cast<diopiTensorHandle_t>(grad_output_tensor_tmp), grad_output_t));
    DIOPI_CALL(permute_to_nhwc(static_cast<diopiTensorHandle_t>(indices_tensor_tmp), indices_t));

    DiopiTensor input_tensor_t(input_t);
    DiopiTensor grad_input_tensor_t(grad_input_t);
    DiopiTensor grad_output_tensor_t(grad_output_t);
    DiopiTensor indices_tensor_t(indices_t);

    std::vector<int> input_dim = getDim(input_tensor_t);
    std::vector<int> grad_input_dim = getDim(grad_input_tensor_t);
    std::vector<int> grad_output_dim = getDim(grad_output_tensor_t);
    std::vector<int> indices_dim = getDim(indices_tensor_t);
    CnnlTensorDesc input_desc;
    CnnlTensorDesc grad_input_desc;
    CnnlTensorDesc grad_output_desc;
    CnnlTensorDesc indices_desc;
    input_desc.set(input_tensor_t, CNNL_LAYOUT_NHWC, input_dim);
    grad_input_desc.set(grad_input_tensor_t, CNNL_LAYOUT_NHWC, grad_input_dim);
    grad_output_desc.set(grad_output_tensor_t, CNNL_LAYOUT_NHWC, grad_output_dim);
    indices_desc.set(indices_tensor_t, CNNL_LAYOUT_NHWC, indices_dim);

    const int64_t kernel_h = kernel_size.data[0];
    const int64_t kernel_w = kernel_size.len == 1 ? kernel_h : kernel_size.data[1];
    int64_t stride_h = 0;
    int64_t stride_w = 0;
    if (stride.len == 0) {
        stride_h = kernel_h;
        stride_w = kernel_w;
    } else {
        stride_h = stride.data[0];
        stride_w = stride.len == 1 ? stride_h : stride.data[1];
    }
    const int64_t pad_h = padding.data[0];
    const int64_t pad_w = padding.len == 1 ? pad_h : padding.data[1];
    const int64_t dilation_0 = dilation.data[0];
    const int64_t dilation_1 = dilation.len == 1 ? dilation_0 : dilation.data[1];

    // calculate padding coefficients
    auto pl = 0, pr = 0, pu = 0, pd = 0;
    pu = pd = pad_h;
    pl = pr = pad_w;
    int height = (grad_output_dim[1] - 1) * stride_h + kernel_h;
    int width = (grad_output_dim[2] - 1) * stride_w + kernel_w;
    if (pad_h + input_dim[1] >= height) {
        pd = 0;
    }
    if (pad_w + input_dim[2] >= width) {
        pr = 0;
    }
    // if ceil_mode is set to true, the pad needs to be filled up.
    if (ceil_mode) {
        pd = height - input_dim[1] - pad_h;
        pr = width - input_dim[2] - pad_w;
    }

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> CnnlPoolDesc;
    cnnlPoolingDescriptor_t pool_desc = CnnlPoolDesc.get();
    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        pool_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernel_h, kernel_w, pu, pd, pl, pr, stride_h, stride_w, dilation_0, dilation_1, ceil_mode));

    DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                       pool_desc,
                                       nullptr,
                                       indices_desc.get(),
                                       indices_tensor_t.data(),
                                       grad_output_desc.get(),
                                       grad_output_tensor_t.data(),
                                       input_desc.get(),
                                       input_tensor_t.data(),
                                       nullptr,
                                       grad_input_desc.get(),
                                       grad_input_tensor_t.data()));

    if (grad_input_tensor_t.shape().size() == 4) {
        std::vector<int64_t> perm_nhwc2nchw{0, 3, 1, 2};
        diopiSize_t nhwc2nchw(perm_nhwc2nchw.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, static_cast<diopiTensorHandle_t>(grad_input_tensor_tmp), grad_input_t, nhwc2nchw));
    } else if (grad_input_tensor_t.shape().size() == 3) {
        std::vector<int64_t> perm_hwc2chw{2, 0, 1};
        diopiSize_t hwc2chw(perm_hwc2chw.data(), 3);
        DIOPI_CALL(diopiPermute(ctx, static_cast<diopiTensorHandle_t>(grad_input_tensor_tmp), grad_input_t, hwc2chw));
    } else {
        DIOPI_CHECK(false, "non-empty 3D or 4D (batch mode) tensor expected for input");
    }
    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, grad_input_tensor_tmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
