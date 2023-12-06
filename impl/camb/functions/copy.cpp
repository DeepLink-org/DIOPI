/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"

using namespace std;
using namespace std::chrono;  // 增加引用空间

namespace impl {
namespace camb {

static bool denseCheck(const DiopiTensor& src) {
    int dim = src.dim();
    std::vector<int> stride(dim, 1);
    std::vector<int> shape(dim, 1);

    for (int i = 0; i < dim; i++) {
        stride[i] = src.stride()[i];
        shape[i] = src.shape()[i];
        if (src.stride()[i] == 0 || src.shape()[i] == 0) {
            return false;
        }
    }

    //这种情况算order存在二义性
    for (int i = 0; i < dim - 1; i++) {
        for (int j = i + 1; j < dim; j++) {
            if (shape[i] == shape[j]) {
                return false;
            }
        }
    }
    std::sort(stride.begin(), stride.end());

    // e.g. shape = 2,3,4,5,stride = 1,3,12,60
    //这种情况肯定是非dense
    if (stride[0] != 1) {
        return false;
    }

    int cur = 1;
    for (int i = 1; i < dim; i++) {
        cur = stride[i] / stride[i - 1];
        if (std::find(shape.begin(), shape.end(), cur) != shape.end()) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

static bool probableMemoryFormat(const DiopiTensor& src, diopiMemoryFormat_t* outMemoryFormat) {
    if (!outMemoryFormat) {
        return src.isContiguous(diopiMemoryFormat_t::Contiguous) || src.isContiguous(diopiMemoryFormat_t::ChannelsLast1d) ||
               src.isContiguous(diopiMemoryFormat_t::ChannelsLast) || src.isContiguous(diopiMemoryFormat_t::ChannelsLast3d);
    }
    if (src.isContiguous(diopiMemoryFormat_t::Contiguous)) {
        *outMemoryFormat = diopiMemoryFormat_t::Contiguous;
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast1d)) {
        *outMemoryFormat = diopiMemoryFormat_t::ChannelsLast1d;
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast)) {
        *outMemoryFormat = diopiMemoryFormat_t::ChannelsLast;
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast3d)) {
        *outMemoryFormat = diopiMemoryFormat_t::ChannelsLast3d;
    } else {
        // memory format not supported.
        return false;
    }
    return true;
}

void removeTheFrontOneInShape(DiopiTensor& src, const std::vector<int64_t>& destShape) {
    int64_t srcDim = src.dim();
    int64_t destDim = destShape.size();
    if (srcDim <= destDim) {
        return;
    }
    // remove the front one in shape
    int64_t diffDim = srcDim - destDim;
    bool canRemoveFlag = true;
    for (int i = 0; i < diffDim; ++i) {
        if (src.shape()[i] != 1) {
            canRemoveFlag = false;
            break;
        }
    }
    int64_t offset = canRemoveFlag ? diffDim : 0;
    std::vector<int64_t> newSrcShape(src.shape().begin() + offset, src.shape().end());
    std::vector<int64_t> newSrcStrides(src.stride().begin() + offset, src.stride().end());
    src.asStrided(newSrcShape, newSrcStrides);
    return;
}

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    if (src == dest) {
        // the same address of pointers, return earlier
        return diopiSuccess;
    }
    DiopiTensor srcTr(src);
    DiopiTensor destTr(dest);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    if (!srcTr.defined()) {
        return diopiSuccess;
    }
    DIOPI_CHECK(destTr.defined(), "dest is not defined but src is defined.")
    if (destTr.numel() == 0) {
        return diopiSuccess;
    }

    // memory format convert if memory format is matched.
    diopiMemoryFormat_t destMemoryFormat;
    // cnnTranspose doesn't support float64 and scalar and contiguousOut only support convertion between the contiguous tensor and the no-contiguous tensor.
    if (srcTr.shape() == destTr.shape() && srcTr.dim() != 0 && srcTr.dtype() != diopi_dtype_float64 && denseCheck(srcTr) &&
        (destTr.isContiguous() && !srcTr.isContiguous())) {
        DiopiTensor destTmpTr = destTr;
        probableMemoryFormat(destTr, &destMemoryFormat);
        if (destTmpTr.dtype() != srcTr.dtype()) {
            destTmpTr = requiresTensor(ctx, destTr.shape(), srcTr.dtype());
        }
        DIOPI_CALL(contiguousOut(ctx, srcTr, destTmpTr, destMemoryFormat));
        if (destTmpTr.dtype() != destTr.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, destTr, destTmpTr));
        }
        return diopiSuccess;
    }

    // Ordinary copy
    // broadcast
    if (srcTr.shape() != destTr.shape()) {
        std::vector<int64_t> destStrides;
        DiopiTensor srcBroadcasted;
        removeTheFrontOneInShape(srcTr, destTr.shape());  // remove this when some ops (max_pool2d etc.) are refactored by right shape.
        if (broadcast(srcTr, destTr.shape(), &srcBroadcasted)) {
            srcTr = srcBroadcasted;
        } else {
            DIOPI_CHECK(false,
                        "can't broadcast because of the mismatched shape, src's shape: (%s), the dest's shape: (%s)",
                        vec2str(srcTr.shape()).c_str(),
                        vec2str(destTr.shape()).c_str());  // return
            return diopiErrorOccurred;
        }
    }

    // data type cast
    if (srcTr.dtype() != destTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, srcTr, destTr.dtype()));
    }
    CnnlTensorDesc inputDesc(destTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(srcTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlCopy(handle, srcDesc.get(), srcTr.data(), inputDesc.get(), destTr.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
