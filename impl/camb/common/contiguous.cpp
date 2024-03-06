/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>
#include <vector>

#include "common.hpp"
namespace impl {
namespace camb {

static diopiError_t transpose(diopiContextHandle_t& ctx, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layoutIn, cnnlTensorLayout_t layoutOut,
                              std::vector<int32_t> order) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inDesc(in, layoutIn);
    CnnlTensorDesc outDesc(out, layoutOut);
    CnnlTransposeDescriptor transDesc(order.size(), order.data());
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();
    DIOPI_CALL_CNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(), in.data(), outDesc.get(), out.data(), workspacePtr, workspaceSize));
    return diopiSuccess;
}

diopiError_t getPermuteOrder(const DiopiTensor& src, std::vector<int32_t>& orderOut, std::vector<int32_t>& reverseOrder) {
    if (src.isContiguous()) {
        orderOut.resize(src.dim());
        for (int i = 0; i < src.dim(); ++i) {
            orderOut[i] = i;
        }
        reverseOrder = orderOut;
        return diopiSuccess;
    }

    std::vector<int64_t> stride = src.stride();
    std::vector<int64_t> shape = src.shape();
    getPermuteOrder(shape, stride, orderOut, reverseOrder);
    return diopiSuccess;
}

diopiError_t getPermuteOrder(std::vector<int64_t>& shape, std::vector<int64_t>& stride, std::vector<int32_t>& orderOut, std::vector<int32_t>& reverseOrder) {
    int dim = shape.size();
    std::vector<int> inputStrides(stride.begin(), stride.end());
    std::vector<int> inputSizes(shape.begin(), shape.end());

    std::vector<std::pair<int, int>> stridesSizes(dim, std::pair<int, int>(1, 1));
    for (int i = 0; i < dim; ++i) {
        stridesSizes[i] = std::pair<int, int>(inputStrides[i], inputSizes[i]);
    }
    orderOut.resize(dim);
    reverseOrder.resize(dim);
    // shape:2,3,4,5 stride:60,1,15,3 -> orderOut: 0,3,1,2, reverseOrder: 0,2,3,1
    sort(stridesSizes.begin(), stridesSizes.end(), [](std::pair<int, int> a, std::pair<int, int> b) { return a.first > b.first; });
    for (int i = 0; i < dim; ++i) {
        auto pair = stridesSizes[i];
        for (int j = 0; j < dim; ++j) {
            if ((pair.first == inputStrides[j]) && (pair.second == inputSizes[j])) {
                reverseOrder[i] = j;
                inputStrides[j] = -1;
                inputSizes[j] = -1;
                break;
            }
        }
    }

    // 反推orderOut
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (reverseOrder[j] == i) {
                orderOut[i] = j;
            }
        }
    }
    return diopiSuccess;
}

diopiError_t calCnnlLayout(diopiMemoryFormat_t memoryFormat, int64_t dim, cnnlTensorLayout_t& cnnlLayout) {
    switch (memoryFormat) {
        case diopiMemoryFormat_t::ChannelsLast1d:
            cnnlLayout = CNNL_LAYOUT_NLC;
        case diopiMemoryFormat_t::ChannelsLast:
            cnnlLayout = CNNL_LAYOUT_NHWC;
            break;
        case diopiMemoryFormat_t::ChannelsLast3d:
            cnnlLayout = CNNL_LAYOUT_NDHWC;
            break;
        case diopiMemoryFormat_t::Contiguous:
            if (dim == 4) {
                cnnlLayout = CNNL_LAYOUT_NCHW;
            } else if (dim == 5) {
                cnnlLayout = CNNL_LAYOUT_NCDHW;
            } else {
                setLastErrorString("memoryFormat (%d) is not matched.", memoryFormat);
                return diopiNoImplement;
            }
            break;
        default:
            setLastErrorString("memoryFormat (%d) is not matched.", memoryFormat);
            return diopiNoImplement;
    }
    return diopiSuccess;
}

std::vector<int64_t> changeVecAccordingToOrder(const std::vector<int64_t> vec, std::vector<int32_t> order) {
    DIOPI_CHECK_ABORT(order.size() == vec.size(), "order's len %ld is not equal vec's len %ld", order.size(), vec.size());
    std::vector<int64_t> newVec(vec.size(), 0);
    int j = 0;
    for (auto i : order) {
        newVec[j++] = vec[i];
    }
    return newVec;
}

std::vector<int64_t> calContiguousStride(std::vector<int64_t> shape) {
    int32_t len = shape.size();
    std::vector<int64_t> stride(len, 1);
    int64_t strideTmp = 1;
    for (int i = len - 1; i > 0; i--) {
        stride[i] = strideTmp;
        strideTmp *= shape[i];
    }
    stride[0] = strideTmp;
    return stride;
}

// change the shape and stride, the stride is incremental.
// order: 0, 2, 3, 1
// shape: 2,3,4,5 stride: 60, 1, 15, 3  -->
// shape: 2,4,5,3 stride: 60, 15, 3, 1
diopiError_t permuteTensor(DiopiTensor& t, const std::vector<int32_t>& order) {
    std::vector<int64_t> newShape = changeVecAccordingToOrder(t.shape(), order);
    std::vector<int64_t> newStride = calContiguousStride(t.shape());
    t.asStrided(newShape, newStride);
    return diopiSuccess;
}

// inplace contiguous, support NCHW <-> NHWC, NCDHW <-> NDHWC  NCL <-> NLC
diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, diopiMemoryFormat_t memoryFormat) {
    if (!denseCheck(src)) {
        DiopiTensor denseOut;
        toDense(ctx, src, denseOut);
        src = denseOut;
        if (memoryFormat == diopiMemoryFormat_t::Preserve) {
            // no need for further permute, if memoryFormat is Preserve.
            return diopiSuccess;
        }
    }

    if (src.isContiguous(memoryFormat)) {
        return diopiSuccess;
    }

    int64_t dim = src.dim();
    DIOPI_CHECK(dim <= 8, "only support less than 8d tensor currently");
    DiopiTensor dest;
    DIOPI_CALL(clone(ctx, src, dest, memoryFormat));
    src = dest;
    return diopiSuccess;
}

diopiError_t permuteCopy(diopiContextHandle_t ctx, DiopiTensor& src, DiopiTensor& dest) {
    // using input permute + output permute + cnnltranspose to copy
    DIOPI_CHECK(src.shape() == dest.shape(), "src's shape should be the same as dest's");
    int64_t dim = src.dim();
    DIOPI_CHECK(dim <= 8, "only support less than 8d tensor currently");
    bool srcIsContiguous = src.isContiguous();
    bool destIsContiguous = dest.isContiguous();
    std::vector<int32_t> inputOrder(dim, 0);
    std::vector<int32_t> inputBackOrder(dim, 0);  // permuteTensor(input,inputBackOrder)->contiguous
    std::vector<int32_t> outputOrder(dim, 0);
    std::vector<int32_t> outputBackOrder(dim, 0);     // permuteTensor(output,outputBackOrder)->contiguous
    std::vector<int32_t> inputToOutputOrder(dim, 0);  // into cnnltranspose

    // input shape:2,3,4,5 stride:60,1,15,3 -> inputBackOrder: 0,2,3,1, inputOrder: 0,3,1,2
    // output shape:2,3,4,5 stride:60,20,1,4 -> outputBackOrder: 0,1,3,2, outputOrder: 0,1,3,2
    // inputToOutputOrder: 0,2,1,3

    getPermuteOrder(src, inputOrder, inputBackOrder);
    getPermuteOrder(dest, outputOrder, outputBackOrder);

    cnnlTensorLayout_t srcLayout = CNNL_LAYOUT_ARRAY;
    cnnlTensorLayout_t destLayout = CNNL_LAYOUT_ARRAY;

    std::vector<int64_t> olderDestStride = dest.stride();
    std::vector<int64_t> olderDestShape = dest.shape();
    std::vector<int64_t> olderSrcStride = src.stride();
    std::vector<int64_t> olderSrcShape = src.shape();

    // permute to get contiguous tensor
    if (!destIsContiguous) {
        DIOPI_CALL(permuteTensor(dest, outputBackOrder));
    }

    if (!srcIsContiguous) {
        DIOPI_CALL(permuteTensor(src, inputBackOrder));
    }

    for (int i = 0; i < dim; ++i) {
        inputToOutputOrder[i] = inputOrder[outputBackOrder[i]];
    }

    DIOPI_CALL(transpose(ctx, src, dest, srcLayout, destLayout, inputToOutputOrder));

    // recovery the shape and strides
    if (!destIsContiguous) {
        dest.asStrided(olderDestShape, olderDestStride);
    }

    if (!srcIsContiguous) {
        src.asStrided(olderSrcShape, olderSrcStride);
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
