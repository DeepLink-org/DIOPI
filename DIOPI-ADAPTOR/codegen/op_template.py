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
#include <iostream>
#include <vector>
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <utils.hpp>

namespace diopiadaptor{

static std::vector<diopiMemoryFormat_t> defaultFormats{diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast};

${cast_strategy}

template<class T, class strategy = NoCast, bool needContiguous = false>
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
    auto memoryFormat = probableMemoryFormat(src);
    bool convertFormat = true;
    for (int i = 0; i < supportMemoryFormat.size(); ++i) {
        if (supportMemoryFormat[i] == memoryFormat) {
            convertFormat = false;
            break;
        }
    }
    bool contiguous = needContiguous && isContiguous(size, stride, memoryFormat);
    int convertType = 0;
    if (!convertFormat) {
        dstStride = stride;
    } else {
        auto strides_v = calcStrides(size.len, size, memoryFormat);
        dstStride.len = strides_v.size();
        dstStride.data = strides_v.data();
    }
    if (convertDtype) {
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &size, &stride, dstDtype, device);
        diopiCastDtype(ctx, tmp, src);
        *dst = tmp;
        convertType = 1;
    } else {
        *dst = src;
    }
    convertType = convertType << 1;
    if (convertFormat || !contiguous) {
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &size, &stride, dstDtype, device);
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

template<class strategy = NoCast, bool isContiguous = false>
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
        convertType_ = castImpl<diopiTensorHandle_t, strategy, isContiguous>(ctx, payload_, &tmp_, supportMemoryFormat);
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