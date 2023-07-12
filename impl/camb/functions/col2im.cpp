/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

extern "C" {
static std::vector<int> getPerm(DiopiTensor tensor, int64_t dim0, int64_t dim1) {
    int inputSize = tensor.shape().size();
    if (dim0 < 0) {
        dim0 = dim0 + inputSize;
    }
    if (dim1 < 0) {
        dim1 = dim1 + inputSize;
    }

    std::vector<int> perms(inputSize);
    std::iota(perms.begin(), perms.end(), 0);
    perms[dim0] = dim1;
    perms[dim1] = dim0;
    return perms;
}

static diopiError_t transposeInternal(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor input, int64_t dim0, int64_t dim1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> cnnlTransposeDesc;
    cnnlTransposeDescriptor_t transposeDesc = cnnlTransposeDesc.get();
    std::vector<int> perms = getPerm(input, dim0, dim1);
    cnnlSetTransposeDescriptor(transposeDesc, perms.size(), perms.data());

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transposeDesc, &workspaceSize);
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    cnnlTranspose_v2(handle, transposeDesc, inputDesc.get(), input.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize);
    return diopiSuccess;
}

diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size,
                         diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor input_col = requiresTensor(ctx, {input_tensor.shape()[0], input_tensor.shape()[2], input_tensor.shape()[1]}, input_tensor.dtype());
    DIOPI_CALL(transposeInternal(ctx, input_col, input_tensor, 1, 2));
    CnnlTensorDesc input_colDesc(input_col, CNNL_LAYOUT_ARRAY);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, input_tensor.dtype()));

    DiopiTensor out_tensor(out);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_NCHW);

    int32_t pad_height = padding.data[0];
    int32_t pad_width = padding.len == 2 ? padding.data[1] : padding.data[0];
    std::vector<int32_t> v_padding = {pad_height, pad_height, pad_width, pad_width};
    int32_t stride_height = stride.data[0];
    int32_t stride_width = stride.len == 2 ? stride.data[1] : stride.data[0];
    std::vector<int32_t> v_stride = {stride_height, stride_width};
    int32_t dilation_height = dilation.data[0];
    int32_t dilation_width = dilation.len == 2 ? dilation.data[1] : dilation.data[0];
    std::vector<int32_t> v_dilation = {dilation_height, dilation_width};
    int32_t kernel_size_height = kernel_size.data[0];
    int32_t kernel_size_width = kernel_size.len == 2 ? kernel_size.data[1] : kernel_size.data[0];
    int32_t output_size_height = output_size.data[0];
    int32_t output_size_width = output_size.len == 2 ? output_size.data[1] : output_size.data[0];

    CnnlTensorDesc weightDesc;
    cnnlTensorDescriptor_t w_desc = weightDesc.get();
    std::vector<int> weight_sizes = {1, 1, kernel_size_height, kernel_size_width};
    std::vector<int> weight_strides = {1, 1, 1, 1};
    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(w_desc, CNNL_LAYOUT_NCHW, dtype, weight_sizes.size(), weight_sizes.data(), weight_strides.data()));

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> ConvDesc;
    cnnlConvolutionDescriptor_t conv_desc = ConvDesc.get();
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(conv_desc, 4, v_padding.data(), v_stride.data(), v_dilation.data(), 1, dtype));

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetCol2ImWorkspaceSize(handle, input_colDesc.get(), outDesc.get(), w_desc, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlCol2Im(handle, input_colDesc.get(), input_col.data(), w_desc, conv_desc, workspace, workspace_size, outDesc.get(), out_tensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
