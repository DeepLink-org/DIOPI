/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {
DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx,
                                      diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size,
                                      diopiSize_t stride,
                                      diopiSize_t padding,
                                      diopiSize_t dilation,
                                      bool ceil_mode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);

    DIOPI_CHECK(input_tensor.dim() == 4, "4D (batch mode) tensor expected for input");

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_NCHW);

    const void* input_ptr = input_tensor.data();
    void* out_ptr = out_tensor.data();

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> CnnlPoolDesc;
    cnnlPoolingDescriptor_t pool_desc = CnnlPoolDesc.get();
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
        int diff_height = (out_tensor.shape()[2] - 1) * stride_h + kernel_h - input_tensor.shape()[2];
        int diff_width = (out_tensor.shape()[3] - 1) * stride_w + kernel_w - input_tensor.shape()[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diff_height > pad_h ? diff_height - pad_h : 0;
        pr = diff_width > pad_w ? diff_width - pad_w : 0;
    }

    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        pool_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernel_h, kernel_w, pu, pd, pl, pr, stride_h, stride_w, dilation_0, dilation_1, ceil_mode));

    std::vector<int8_t> extra_host_input;
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWorkspaceSize(handle, CNNL_POOLING_MAX, out_tensor.shape()[3], input_tensor.shape()[2], &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    const void* alpha = nullptr;
    const void* beta = nullptr;
    DIOPI_CALLCNNL(cnnlPoolingForward(handle, pool_desc, alpha, input_desc.get(), input_ptr, beta, out_desc.get(), out_ptr, workspace, workspace_size));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t out,
                                                 diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input,
                                                 diopiSize_t kernel_size,
                                                 diopiSize_t stride,
                                                 diopiSize_t padding,
                                                 diopiSize_t dilation,
                                                 bool ceil_mode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);
    auto indices_tensor = DiopiTensor(indices);

    DIOPI_CHECK(input_tensor.dim() == 4, "4D (batch mode) tensor expected for input");

    cnnlDataType_t input_dtype;
    cnnlDataType_t indices_dtype_ori;
    cnnlDataType_t indices_dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&input_dtype, input_tensor.dtype()));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&indices_dtype_ori, indices_tensor.dtype()));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&indices_dtype, indices_tensor.dtype()));
    if (input_dtype == CNNL_DTYPE_HALF) {
        indices_dtype = CNNL_DTYPE_INT16;
    }

    diopiSize_t indices_shape;
    DIOPI_CALL(diopiGetTensorShape(indices, &indices_shape));
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_NCHW);
    CnnlTensorDesc indices_desc(indices_tensor, CNNL_LAYOUT_NCHW);

    const void* input_ptr = input_tensor.data();
    void* out_ptr = out_tensor.data();
    void* indices_ptr = indices_tensor.data();

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> CnnlPoolDesc;
    cnnlPoolingDescriptor_t pool_desc = CnnlPoolDesc.get();
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
        int diff_height = (out_tensor.shape()[2] - 1) * stride_h + kernel_h - input_tensor.shape()[2];
        int diff_width = (out_tensor.shape()[3] - 1) * stride_w + kernel_w - input_tensor.shape()[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diff_height > pad_h ? diff_height - pad_h : 0;
        pr = diff_width > pad_w ? diff_width - pad_w : 0;
    }

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

    std::vector<int8_t> extra_host_input;
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetPoolingWithIndexWorkspaceSize(handle, input_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    const void* alpha = nullptr;
    const void* beta = nullptr;
    diopiTensorHandle_t indices32_tmp;
    DIOPI_CALL(diopiRequireTensor(ctx, &indices32_tmp, &indices_shape, nullptr, diopi_dtype_int32, diopi_device));
    auto indices32_tmp_tensor = DiopiTensor(indices32_tmp);
    CnnlTensorDesc indices32_tmp_desc(indices32_tmp_tensor, CNNL_LAYOUT_NCHW);
    void* indices32_tmp_ptr = indices32_tmp_tensor.data();

    if (indices_dtype == CNNL_DTYPE_INT64) {
        DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(handle,
                                                   pool_desc,
                                                   alpha,
                                                   input_desc.get(),
                                                   input_ptr,
                                                   beta,
                                                   out_desc.get(),
                                                   out_ptr,
                                                   indices32_tmp_desc.get(),
                                                   indices32_tmp_ptr,
                                                   workspace,
                                                   workspace_size));
        DIOPI_CHECKCNNL(cnnlCastDataType(handle, indices32_tmp_desc.get(), indices32_tmp_ptr, CNNL_CAST_INT32_TO_INT64, indices_desc.get(), indices_ptr));
    } else if ((indices_dtype == indices_dtype_ori) && (indices_dtype_ori == CNNL_DTYPE_INT32)) {
        DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(
            handle, pool_desc, alpha, input_desc.get(), input_ptr, beta, out_desc.get(), out_ptr, indices_desc.get(), indices_ptr, workspace, workspace_size));
    } else if ((indices_dtype == CNNL_DTYPE_INT16) && (indices_dtype_ori == CNNL_DTYPE_INT32)) {
        diopiTensorHandle_t indices16_tmp;
        DIOPI_CALL(diopiRequireTensor(ctx, &indices16_tmp, &indices_shape, nullptr, diopi_dtype_int16, diopi_device));
        auto indices16_tmp_tensor = DiopiTensor(indices16_tmp);
        CnnlTensorDesc indices16_tmp_desc(indices16_tmp_tensor, CNNL_LAYOUT_NCHW);
        void* indices16_tmp_ptr = indices16_tmp_tensor.data();
        DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(handle,
                                                   pool_desc,
                                                   alpha,
                                                   input_desc.get(),
                                                   input_ptr,
                                                   beta,
                                                   out_desc.get(),
                                                   out_ptr,
                                                   indices16_tmp_desc.get(),
                                                   indices16_tmp_ptr,
                                                   workspace,
                                                   workspace_size));
        DIOPI_CHECKCNNL(cnnlCastDataType(handle, indices16_tmp_desc.get(), indices16_tmp_ptr, CNNL_CAST_INT16_TO_INT32, indices_desc.get(), indices_ptr));
    } else {
        diopiTensorHandle_t indices16_tmp;
        DIOPI_CALL(diopiRequireTensor(ctx, &indices16_tmp, &indices_shape, nullptr, diopi_dtype_int16, diopi_device));
        auto indices16_tmp_tensor = DiopiTensor(indices16_tmp);
        CnnlTensorDesc indices16_tmp_desc(indices16_tmp_tensor, CNNL_LAYOUT_NCHW);
        void* indices16_tmp_ptr = indices16_tmp_tensor.data();
        DIOPI_CALLCNNL(cnnlPoolingForwardWithIndex(handle,
                                                   pool_desc,
                                                   alpha,
                                                   input_desc.get(),
                                                   input_ptr,
                                                   beta,
                                                   out_desc.get(),
                                                   out_ptr,
                                                   indices16_tmp_desc.get(),
                                                   indices16_tmp_ptr,
                                                   workspace,
                                                   workspace_size));
        DIOPI_CHECKCNNL(
            cnnlCastDataType(handle, indices16_tmp_desc.get(), indices16_tmp_ptr, CNNL_CAST_INT16_TO_INT32, indices32_tmp_desc.get(), indices32_tmp_ptr));
        DIOPI_CHECKCNNL(cnnlCastDataType(handle, indices32_tmp_desc.get(), indices32_tmp_ptr, CNNL_CAST_INT32_TO_INT64, indices_desc.get(), indices_ptr));
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx,
                                              diopiTensorHandle_t grad_input,
                                              diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input,
                                              diopiSize_t kernel_size,
                                              diopiSize_t stride,
                                              diopiSize_t padding,
                                              diopiSize_t dilation,
                                              bool ceil_mode,
                                              diopiConstTensorHandle_t indices) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_output_tensor = DiopiTensor(grad_output);
    auto indices_tensor = DiopiTensor(indices);
    DIOPI_CHECK(input_tensor.dim() == 4, "4D (batch mode) tensor expected for input");

    cnnlDataType_t grad_output_dtype;
    cnnlDataType_t indices_dtype_ori;
    cnnlDataType_t indices_dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&grad_output_dtype, grad_output_tensor.dtype()));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&indices_dtype_ori, indices_tensor.dtype()));
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&indices_dtype, indices_tensor.dtype()));
    if (grad_output_dtype == CNNL_DTYPE_HALF) {
        indices_dtype = CNNL_DTYPE_INT16;
    }

    diopiTensorHandle_t input_t;
    diopiTensorHandle_t grad_input_t;
    diopiTensorHandle_t grad_output_t;
    diopiTensorHandle_t indices_t;

    auto permute_to_nhwc = [&](auto src, auto& dst) {
        std::vector<int64_t> axis{0, 2, 3, 1};
        auto src_tensor = DiopiTensor(src);
        std::vector<int64_t> src_shape_t_64(src_tensor.shape().size());
        for (int i = 0; i < src_tensor.shape().size(); ++i) {
            src_shape_t_64[i] = src_tensor.shape()[axis[i]];
        }
        diopiSize_t src_t_shape(src_shape_t_64.data(), src_shape_t_64.size());
        DIOPI_CALL(diopiRequireTensor(ctx, &dst, &src_t_shape, nullptr, src_tensor.dtype(), diopi_device));
        diopiSize_t nchw2nhwc(axis.data(), 4);
        DIOPI_CALL(diopiPermute(ctx, dst, src, nchw2nhwc));
        return diopiSuccess;
    };

    DIOPI_CALL(permute_to_nhwc(input, input_t));
    DIOPI_CALL(permute_to_nhwc(grad_input, grad_input_t));
    DIOPI_CALL(permute_to_nhwc(grad_output, grad_output_t));
    DIOPI_CALL(permute_to_nhwc(indices, indices_t));

    auto input_tensor_t = DiopiTensor(input_t);
    auto grad_input_tensor_t = DiopiTensor(grad_input_t);
    auto grad_output_tensor_t = DiopiTensor(grad_output_t);
    auto indices_tensor_t = DiopiTensor(indices_t);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc grad_input_desc(grad_input_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc grad_output_desc(grad_output_tensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc indices_t_desc(indices_tensor, CNNL_LAYOUT_NHWC);

    const void* input_ptr = input_tensor_t.data();
    void* grad_input_ptr = grad_input_tensor_t.data();
    const void* grad_output_ptr = grad_output_tensor_t.data();

    CnnlResourceGuard<cnnlPoolingDescriptor_t, cnnlCreatePoolingDescriptor, cnnlDestroyPoolingDescriptor> CnnlPoolDesc;
    cnnlPoolingDescriptor_t pool_desc = CnnlPoolDesc.get();
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
        int diff_height = (grad_output_tensor.shape()[2] - 1) * stride_h + kernel_h - input_tensor.shape()[2];
        int diff_width = (grad_output_tensor.shape()[3] - 1) * stride_w + kernel_w - input_tensor.shape()[3];
        // If ceil_mode is set to true, the pad needs to be filled up.
        // If the offset pad is redundant, it will be removed.
        pd = diff_height > pad_h ? diff_height - pad_h : 0;
        pr = diff_width > pad_w ? diff_width - pad_w : 0;
    }

    DIOPI_CALLCNNL(cnnlSetPooling2dDescriptor_v2(
        pool_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernel_h, kernel_w, pu, pd, pl, pr, stride_h, stride_w, dilation_0, dilation_1, ceil_mode));
    const void* alpha = nullptr;
    const void* beta = nullptr;

    diopiSize_t indices_t_shape;
    DIOPI_CALL(diopiGetTensorShape(indices_t, &indices_t_shape));
    std::vector<int> indices_shape_{indices_t_shape.data, indices_t_shape.data + indices_t_shape.len};

    diopiTensorHandle_t indices32_cast;
    DIOPI_CALL(diopiRequireTensor(ctx, &indices32_cast, &indices_t_shape, nullptr, diopi_dtype_int32, diopi_device));
    auto indices32_cast_tensor = DiopiTensor(indices32_cast);
    CnnlTensorDesc indices32_cast_desc;
    DIOPI_CALL(indices32_cast_desc.set(indices32_cast_tensor, CNNL_LAYOUT_NHWC, indices_shape_));

    diopiTensorHandle_t indices16_cast;
    DIOPI_CALL(diopiRequireTensor(ctx, &indices16_cast, &indices_t_shape, nullptr, diopi_dtype_int16, diopi_device));
    auto indices16_cast_tensor = DiopiTensor(indices16_cast);
    CnnlTensorDesc indices16_cast_desc;
    DIOPI_CALL(indices16_cast_desc.set(indices16_cast_tensor, CNNL_LAYOUT_NHWC, indices_shape_));

    if (indices_dtype == CNNL_DTYPE_INT64) {
        DIOPI_CHECKCNNL(cnnlCastDataType(
            handle, indices_t_desc.get(), indices_tensor_t.data(), CNNL_CAST_INT64_TO_INT32, indices32_cast_desc.get(), indices32_cast_tensor.data()));
        DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                           pool_desc,
                                           alpha,
                                           indices32_cast_desc.get(),
                                           indices32_cast_tensor.data(),
                                           grad_output_desc.get(),
                                           grad_output_ptr,
                                           input_desc.get(),
                                           input_ptr,
                                           beta,
                                           grad_input_desc.get(),
                                           grad_input_ptr));
    } else if ((indices_dtype == indices_dtype_ori) && (indices_dtype_ori == CNNL_DTYPE_INT32)) {
        DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                           pool_desc,
                                           alpha,
                                           indices_t_desc.get(),
                                           indices_tensor_t.data(),
                                           grad_output_desc.get(),
                                           grad_output_ptr,
                                           input_desc.get(),
                                           input_ptr,
                                           beta,
                                           grad_input_desc.get(),
                                           grad_input_ptr));
    } else if ((indices_dtype == CNNL_DTYPE_INT16) && (indices_dtype_ori == CNNL_DTYPE_INT32)) {
        DIOPI_CHECKCNNL(cnnlCastDataType(
            handle, indices_t_desc.get(), indices_tensor_t.data(), CNNL_CAST_INT32_TO_INT16, indices16_cast_desc.get(), indices16_cast_tensor.data()));
        DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                           pool_desc,
                                           alpha,
                                           indices16_cast_desc.get(),
                                           indices16_cast_tensor.data(),
                                           grad_output_desc.get(),
                                           grad_output_ptr,
                                           input_desc.get(),
                                           input_ptr,
                                           beta,
                                           grad_input_desc.get(),
                                           grad_input_ptr));
    } else {
        DIOPI_CHECKCNNL(cnnlCastDataType(
            handle, indices_t_desc.get(), indices_tensor_t.data(), CNNL_CAST_INT64_TO_INT32, indices32_cast_desc.get(), indices32_cast_tensor.data()));
        DIOPI_CHECKCNNL(cnnlCastDataType(handle,
                                         indices32_cast_desc.get(),
                                         indices32_cast_tensor.data(),
                                         CNNL_CAST_INT32_TO_INT16,
                                         indices16_cast_desc.get(),
                                         indices16_cast_tensor.data()));
        DIOPI_CALLCNNL(cnnlPoolingBackward(handle,
                                           pool_desc,
                                           alpha,
                                           indices16_cast_desc.get(),
                                           indices16_cast_tensor.data(),
                                           grad_output_desc.get(),
                                           grad_output_ptr,
                                           input_desc.get(),
                                           input_ptr,
                                           beta,
                                           grad_input_desc.get(),
                                           grad_input_ptr));
    }

    std::vector<int64_t> perm_nhwc2nchw{0, 3, 1, 2};
    diopiSize_t nhwc2nchw(perm_nhwc2nchw.data(), 4);
    DIOPI_CALL(diopiPermute(ctx, grad_input, grad_input_t, nhwc2nchw));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
