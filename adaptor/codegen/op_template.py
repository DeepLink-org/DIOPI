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

inline bool isContiguous(diopiSize_t size, diopiSize_t stride_diopi, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    auto dim = size.len;
    auto shape = size.data;
    auto strides = stride_diopi.data;
    int64_t stride = 1;

    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = dim - 1; i >= 0; i--) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        if (dim != 4) return false;
        for (auto& i : {1, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                // shape_d != 1 help dealing with shape like [2, 2048, 1, 1]
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        if (dim != 5) return false;
        for (auto& i : {1, 4, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    }
    else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        if (dim != 3) return false;
        for (auto& i : {1, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    }
    return true;
}

static std::vector<diopiMemoryFormat_t> defaultFormats{};

${cast_strategy}

diopiMemoryFormat_t getTargetMemoryFormat(int ndims, std::vector<diopiMemoryFormat_t> supportMemoryFormats) {
    switch (ndims) {
        case 1:
        case 2:
            for (auto i : supportMemoryFormats) {
                if (i == diopiMemoryFormat_t::Contiguous) {
                    return i;
                }
            };
            break;
        case 3: {
            for (auto i : supportMemoryFormats) {
                if (i == diopiMemoryFormat_t::ChannelsLast1d || i == diopiMemoryFormat_t::Contiguous) {
                    return i;
                }
            }
            break;
        }
        case 4: {
            for (auto i : supportMemoryFormats){
                if (i == diopiMemoryFormat_t::ChannelsLast || i == diopiMemoryFormat_t::Contiguous) {
                    return i;
                }
            }
            break;
        }
        case 5: {
            for (auto i : supportMemoryFormats){
                if (i == diopiMemoryFormat_t::ChannelsLast3d || i == diopiMemoryFormat_t::Contiguous) {
                    return i;
                }
            }
            break;
        }
        default: {
            return diopiMemoryFormat_t::Contiguous;
        }
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

template<class T, class strategy = NoCast>
inline int requireImpl(diopiContextHandle_t ctx, T src, T* dst,
                    std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats) {
    if (!src) {
        *dst = src;
        return 0;
    }
    diopiDtype_t srcDtype, dstDtype;
    diopiGetTensorDtype(src, &srcDtype);
    diopiSize_t size, stride;
    diopiGetTensorShape(src, &size);
    diopiGetTensorStride(src, &stride);
    diopiDevice_t device;
    diopiGetTensorDevice(src, &device);

    bool convertDtype = strategy::getDstDtype(srcDtype, dstDtype);
    bool convertFormat = true;
    for (int i = 0; i < supportMemoryFormat.size(); ++i) {
        if (isContiguous(size, stride, supportMemoryFormat[i])) {
            convertFormat = false;
            break;
        }
    }
    diopiSize_t dstStride = stride;
    std::vector<int64_t> strides_v;
    if (convertFormat) {
        diopiMemoryFormat_t memoryFormat = getTargetMemoryFormat(size.len, supportMemoryFormat);
        strides_v = calcStrides(size.len, size, memoryFormat);
        dstStride.len = strides_v.size();
        dstStride.data = strides_v.data();
    }
    int convertType = (convertFormat ? 1 : 0) | ((convertDtype ? 1 : 0) << 1);
    if (convertType) {
        diopiRequireTensor(ctx, dst, &size, &dstStride, dstDtype, device);
    } else {
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
    
    // forbid copy/move construct/assignment
    DiopiTensorWrapper(const DiopiTensorWrapper&) = delete;
    DiopiTensorWrapper& operator=(const DiopiTensorWrapper&) = delete;
    DiopiTensorWrapper(DiopiTensorWrapper&&) = delete;
    DiopiTensorWrapper& operator=(DiopiTensorWrapper&&) = delete;

private:
    diopiContextHandle_t ctx_;
    diopiTensorHandle_t payload_;
    diopiTensorHandle_t tmp_ = nullptr;

    // 0 means no change
    // 1 means changes in memoryformat
    // 2 means changes in dtype
    // 3 means changes in dtype and memoryformat
    int convertType_ = 0;

public:
    DiopiTensorWrapper(diopiContextHandle_t ctx, diopiTensorHandle_t payload,
                       std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats, bool inp = false)
                       : ctx_(ctx)
                       , payload_(payload) {
        if (inp) {
            convertType_ = castImpl<diopiTensorHandle_t, strategy>(ctx, payload_, &tmp_, supportMemoryFormat);
        } else {
            convertType_ = requireImpl<diopiTensorHandle_t, strategy>(ctx, payload_, &tmp_, supportMemoryFormat);
        }
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