/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

template <typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T> vec) {
    for (auto i : vec) {
        out << i << " ";
    }
    return out;
}

static bool probableMemoryFormat(const DiopiTensor& src, diopiMemoryFormat_t& outMemoryFormat) {
    if (src.isContiguous(diopiMemoryFormat_t::Contiguous)) {
        outMemoryFormat = diopiMemoryFormat_t::Contiguous;
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast1d)) {
        outMemoryFormat = diopiMemoryFormat_t::ChannelsLast1d;
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast)) {
        outMemoryFormat = diopiMemoryFormat_t::ChannelsLast;
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast3d)) {
        outMemoryFormat = diopiMemoryFormat_t::ChannelsLast3d;
    } else {
        // memory format not supported.
        return false;
    }
    return true;
}

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    if (src == dest) {
        // the same address of pointers, return earlier
        return diopiSuccess;
    }
    DiopiTensor srcTr(src);
    DiopiTensor destTr(dest);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // data type cast
    if (srcTr.dtype() != destTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, srcTr, destTr.dtype()));
    }

    // memory format convert if memory format is matched.
    diopiMemoryFormat_t destMemoryFormat;
    if (srcTr.shape() == destTr.shape() && probableMemoryFormat(destTr, destMemoryFormat) && (srcTr.isContiguous() || destTr.isContiguous())) {
        DIOPI_CALL(contiguousOut(ctx, srcTr, destTr, destMemoryFormat));
        return diopiSuccess;
    }
    // Ordinary copy
    // broadcast
    if (srcTr.shape() != destTr.shape()) {
        std::vector<int64_t> destStrides;
        DiopiTensor srcBroadcasted;
        if (broadcast1(srcTr, destTr.shape(), &srcBroadcasted)) {
            srcTr = srcBroadcasted;
        } else {
            DIOPI_CHECK(false, "copy failed because the shapes are not match")
            std::cout << "src's shape: " << srcTr.shape() << ", dest's shape: " << destTr.shape() << std::endl;
            return diopiErrorOccurred;
        }
    }

    CnnlTensorDesc inputDesc(destTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(srcTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCopy(handle, srcDesc.get(), srcTr.data(), inputDesc.get(), destTr.data()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
