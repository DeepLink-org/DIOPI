# Copyright (c) 2023, DeepLink.
from code_template import CodeTemplate

class OpTemplate(object):
    operators_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_ADAPTOR_HPP_
#define DIOPI_ADAPTOR_HPP_
#include <cassert>
#include <iostream>
#include <vector>
#include <diopi/diopirt.h>
#include <diopi/functions.h>

namespace diopiadaptor{

inline std::vector<int64_t> calcStrides(int ndims, diopiSize_t size, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    std::vector<int64_t> strides;
    strides.resize(ndims);
    int64_t st = 1;
    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = ndims; i > 0; --i) {
            strides[i - 1] = st;
            if (size.data[i - 1] == 0) continue;
            if (size.data[i - 1] == -1) st = -1;
            if (st != -1) st *= size.data[i - 1];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        for (auto k : {1, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) st *= size.data[k];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        for (auto k : {1, 4, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }

    }
    else if(format == diopiMemoryFormat_t::ChannelsLast1d){
        for (auto k : {1, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }
    }
    else {
        // PARROTS_THROW(InvalidArgs) <<
        //         "Invalid MemoryFormat " << memoryFormatName(format);
    }
    return strides;
}

inline bool isLikeChannelsLast(diopiConstTensorHandle_t tensor, bool checkContiguous, diopiMemoryFormat_t format = diopiMemoryFormat_t::ChannelsLast) {
    diopiSize_t shape, stride;
    diopiGetTensorShape(tensor, &shape);
    diopiGetTensorStride(tensor, &stride);
    if (shape.len != 4) return false;
    int64_t totalSize = 1;
    for (int64_t i = 0; i < shape.len; ++i) {
        totalSize *= shape.data[i];
    }
    if (totalSize == 0) return false;
    if (stride.data[0] == stride.data[1]) return false;
    if (checkContiguous) {
        auto realStride = calcStrides(shape.len, shape, format);
        for (int i = 0; i < stride.len; ++i) {
            if (i >= realStride.size() || realStride[i] != stride.data[i]) {
                return false;
            }
        }
        return true;
    } else {
        int64_t st = 1;
        std::vector<int> orders;
        if (format == diopiMemoryFormat_t::ChannelsLast)
            orders = {1, 3, 2, 0};
        else if (format == diopiMemoryFormat_t::ChannelsLast3d)
            orders = {1, 4, 3, 2, 0};
        for (auto k : orders) {
            if (stride.data[k] < st) return false;
            st = stride.data[k] * shape.data[k];
        }
        return true;
    }
}

inline diopiMemoryFormat_t probableMemoryFormat(diopiConstTensorHandle_t tensor, bool exactMatch = false) {
    return isLikeChannelsLast(tensor, exactMatch) ? diopiMemoryFormat_t::ChannelsLast
        : (isLikeChannelsLast(tensor, exactMatch, diopiMemoryFormat_t::ChannelsLast3d) ? diopiMemoryFormat_t::ChannelsLast3d
        : diopiMemoryFormat_t::Contiguous);
}

inline bool isContiguous(diopiSize_t size, diopiSize_t stride, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    int64_t totalSize = 1;
    for (int64_t i = 0; i < size.len; ++i) {
        totalSize *= size.data[i];
    }
    if (totalSize == 0) return false;
    if (format == diopiMemoryFormat_t::ChannelsLast && size.len != 4) return false;
    auto st = calcStrides(size.len, size, format);
    for (int64_t i = 0; i < size.len; ++i) {
        if (st[i] != stride.data[i] && size.data[i] > 1) return false;
    }
    return true;
}

static std::vector<diopiMemoryFormat_t> defaultFormats{};

${cast_strategy}

diopiMemoryFormat_t getTargetMemoryFormat(int ndims, std::vector<diopiMemoryFormat_t> supportMemoryFormats) {
    if (ndims == 3) {
        for (auto i : supportMemoryFormats) {
            if (i == diopiMemoryFormat_t::ChannelsLast1d || i == diopiMemoryFormat_t::Contiguous) {
                return i;
            }
        }
    }
    else if(ndims == 4) {
        for (auto i : supportMemoryFormats){
            if (i == diopiMemoryFormat_t::ChannelsLast || i == diopiMemoryFormat_t::Contiguous) {
                return i;
            }
        }
    }
    else if(ndims == 5) {
        for (auto i : supportMemoryFormats){
            if (i == diopiMemoryFormat_t::ChannelsLast3d || i == diopiMemoryFormat_t::Contiguous) {
                return i;
            }
        }
    }
    else {
        assert(false && "ndims not supported.");
    }
}

template<class T, class strategy = NoCast>
inline int castImpl(diopiContextHandle_t ctx, T src, T* dst,
                    std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats) {
    if (src == nullptr || src == 0) {
        *dst = src;
        return 0;
    }
    diopiDtype_t srcDtype, dstDtype;
    diopiGetTensorDtype(src, &srcDtype);
    diopiSize_t size, stride, dstStride;
    diopiGetTensorShape(src, &size);
    diopiGetTensorStride(src, &stride);
    diopiDevice_t device;
    diopiGetTensorDevice(src, &device);

    bool convertDtype = strategy::getDstDtype(srcDtype, dstDtype);
    bool convertFormat = true;
    if (supportMemoryFormat.size() == 0) {
        convertFormat = false;
    } else {
        for (int i = 0; i < supportMemoryFormat.size(); ++i) {
            if (isContiguous(size, stride, supportMemoryFormat[i])) {
                convertFormat = false;
                break;
            }
        }
    }
    int convertType = 0;
    std::vector<int64_t> strides_v;
    if (!convertFormat) {
        dstStride = stride;
    } else {
        diopiMemoryFormat_t memoryFormat = getTargetMemoryFormat(size.len, supportMemoryFormat);
        strides_v = calcStrides(size.len, size, memoryFormat);
        dstStride.len = strides_v.size();
        dstStride.data = strides_v.data();
    }
    if (convertDtype) {
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &size, &dstStride, dstDtype, device);
        diopiCastDtype(ctx, tmp, src);
        *dst = tmp;
        convertType = 1;
    } else {
        *dst = src;
    }
    convertType = convertType << 1;
    if (convertFormat) {
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &size, &dstStride, dstDtype, device);
        diopiCopyInp(ctx, *dst, tmp);
        *dst = tmp;
        convertType = convertType + 1;
    }
    if (convertType == 0) {
        *dst = src;
    }
    return convertType;
}

template <typename Adaptor, typename... Args>
void dispatch_diopi(diopiContextHandle_t ctx, Args&&... args) {
    auto adaptor = Adaptor();
    adaptor(ctx, std::forward<Args>(args)...);
}

template<class strategy = NoCast>
class DiopiTensorWrapper {
private:
    diopiContextHandle_t ctx_;
    diopiTensorHandle_t payload_;
    diopiTensorHandle_t tmp_ = nullptr;
    int convertType_ = 0;

public:
    DiopiTensorWrapper(diopiContextHandle_t ctx, diopiTensorHandle_t payload,
                       std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats)
                       : ctx_(ctx)
                       , payload_(payload) {
        convertType_ = castImpl<diopiTensorHandle_t, strategy>(ctx, payload_, &tmp_, supportMemoryFormat);
    }

    ~DiopiTensorWrapper() {
        if (convertType_ == 0) {
            if (tmp_) {
                payload_ = tmp_;
            }
            return;
        }
        if (convertType_ == 1){
            diopiCopyInp(ctx_, tmp_, payload_);
        } else if (convertType_ == 2) {
            diopiCastDtype(ctx_, payload_, tmp_);
        } else {
            diopiDtype_t dtype;
            diopiGetTensorDtype(tmp_, &dtype);
            diopiSize_t size, stride, dstStride;
            diopiGetTensorShape(payload_, &size);
            diopiGetTensorStride(payload_, &stride);
            diopiDevice_t device;
            diopiGetTensorDevice(payload_, &device);
            diopiTensorHandle_t tmp = nullptr;
            diopiRequireTensor(ctx_, &tmp, &size, &stride, dtype, device);
            diopiCopyInp(ctx_, tmp_, tmp);
            diopiCastDtype(ctx_, payload_, tmp);
        }
    }

public:
    operator diopiTensorHandle_t() {
        return tmp_;
    }
};

${adaptors}

}
# endif // DIOPI_ADAPTOR_HPP
""")

    adaptor_template = CodeTemplate("""\
inline diopiError_t diopi${op_name}(${attrs}) {
    ${new_input}
    ${cast_input}
    ${cast_output}
    return ::${call_func}
}

""")

    cast_strategy_template = CodeTemplate("""\
class ${cast_name} {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            ${cases}
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

""")