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
#include <algorithm>
#include <vector>
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <chrono>
#include <fstream>
#include <ostream>
#include <stdlib.h>

namespace diopiadaptor {

class TimeElapsedRecord : public std::ofstream {
public:
    TimeElapsedRecord(const char* fileName):std::ofstream(fileName, std::ios::out | std::ios::trunc), enableTiming_(false) {
        if (getenv("DIOPI_ENABLE_TIMING")){
            enableTiming_ = true;
        }
    }
    bool isEnableTiming(){
        return enableTiming_;
    }
private:
    bool enableTiming_;
};


class TimeElapsed{
public:
    TimeElapsed(const char* opName):opName_(opName){
        if(timeElapsedRecord_.isEnableTiming()){
            start_ = std::chrono::steady_clock::now();
        }
    }
    ~TimeElapsed(){
        if(timeElapsedRecord_.isEnableTiming()){
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start_;  // ms
            double elapsedTime = elapsed.count();
            timeElapsedRecord_ << opName_ << ": " << elapsedTime << "ms" << std::endl;
        }
    }
private:
    const char* opName_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
    static TimeElapsedRecord timeElapsedRecord_;
};

TimeElapsedRecord TimeElapsed::timeElapsedRecord_("op_time.dat");

class ConvertType {
public:
    ConvertType(): val_(0) {};
    void setDtypeConverted() {
        val_ |= 0x1u;
    }
    void setMemoryFormatConverted(){
        val_ |= 0x2u;
    }
    bool isDtypeConverted(){
        return val_ & 0x1u;
    }
    bool isMemoryFormatConverted(){
        return val_ & 0x2u;
    }
    bool isConverted(){
        return static_cast<bool>(val_);
    }

private:
    unsigned char val_;

};

inline std::vector<int64_t> calcStrides(diopiSize_t size, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    size_t ndims = size.len;
    std::vector<int64_t> strides(ndims);
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
        auto realStride = calcStrides(shape, format);
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

// diopiMemoryFormat_t getTargetMemoryFormat(int ndims, std::vector<diopiMemoryFormat_t> supportMemoryFormats) {
//     switch (ndims) {
//         case 1:
//         case 2:
//             for (auto i : supportMemoryFormats) {
//                 if (i == diopiMemoryFormat_t::Contiguous) {
//                     return i;
//                 }
//             };
//             break;
//         case 3: {
//             for (auto i : supportMemoryFormats) {
//                 if (i == diopiMemoryFormat_t::ChannelsLast1d || i == diopiMemoryFormat_t::Contiguous) {
//                     return i;
//                 }
//             }
//             break;
//         }
//         case 4: {
//             for (auto i : supportMemoryFormats){
//                 if (i == diopiMemoryFormat_t::ChannelsLast || i == diopiMemoryFormat_t::Contiguous) {
//                     return i;
//                 }
//             }
//             break;
//         }
//         case 5: {
//             for (auto i : supportMemoryFormats){
//                 if (i == diopiMemoryFormat_t::ChannelsLast3d || i == diopiMemoryFormat_t::Contiguous) {
//                     return i;
//                 }
//             }
//             break;
//         }
//         default: {
//             return diopiMemoryFormat_t::Contiguous;
//         }
//     }
// }
inline std::vector<diopiMemoryFormat_t> matchMemoryFormatBySize(size_t sizeLen){
    std::vector<diopiMemoryFormat_t> matchedMemoryFormat;
    switch(sizeLen){
        case 1:
        case 2:
            return {diopiMemoryFormat_t::Contiguous};
        case 3:
            return {diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast1d};
        case 4:
            return {diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast};
        case 5:
            return {diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast3d};
        default:
            return {diopiMemoryFormat_t::Contiguous};
    }
}

inline std::vector<diopiMemoryFormat_t> setIntersection(std::vector<diopiMemoryFormat_t> matchedMemoryFormat, std::vector<diopiMemoryFormat_t> supportMemoryFormat){
    // A small number of elements are suitable for this method getting intersection
    std::vector<diopiMemoryFormat_t>  ret;
    for(auto i : matchedMemoryFormat){
        if(std::find(supportMemoryFormat.begin(), supportMemoryFormat.end(), i) != std::end(supportMemoryFormat)){
            ret.push_back(i);
        }
    }
    return ret;
}

template <typename T>
struct RemoveConst{
    using type = T;
};

template <>
struct RemoveConst<diopiConstTensorHandle_t> {
    using type = diopiTensorHandle_t;
};

std::vector<diopiMemoryFormat_t> obtainTargetMemoryFormats(size_t shapeLen, std::vector<diopiMemoryFormat_t> supportMemoryFormats){
    std::vector<diopiMemoryFormat_t> matchedMemoryFormat = matchMemoryFormatBySize(shapeLen);
    supportMemoryFormats = setIntersection(matchedMemoryFormat, supportMemoryFormats);
    return supportMemoryFormats;
}

template<class T, class strategy = NoCast>
ConvertType castImpl(diopiContextHandle_t ctx, T src, T* dst,
                     std::vector<diopiMemoryFormat_t> supportMemoryFormats = defaultFormats) {
    ConvertType convertType;
    if (!src) {
        *dst = src;
        return convertType;
    }
    diopiDtype_t srcDtype, dstDtype;
    diopiGetTensorDtype(src, &srcDtype);
    diopiSize_t srcSize, srcStride;
    diopiGetTensorShape(src, &srcSize);
    diopiGetTensorStride(src, &srcStride);
    strategy::getDstDtype(srcDtype, dstDtype);
    std::vector<diopiMemoryFormat_t> targetMemoryFormats = obtainTargetMemoryFormats(srcSize.len, supportMemoryFormats);
    diopiTensorHandle_t memoryFormatedTensor = nullptr;

    // convertDtype

    diopiDevice_t device;
    diopiGetTensorDevice(src, &device);
    diopiTensorHandle_t tmp0 = nullptr;
    bool needConvertDtype = srcDtype != dstDtype;
    if(needConvertDtype){
        diopiRequireTensor(ctx, &tmp0, &srcSize, &srcStride, dstDtype, device);
        diopiCastDtype(ctx, tmp0, src);
        convertType.setDtypeConverted();
    }
    else {
        tmp0 = const_cast<typename RemoveConst<T>::type>(src);
    }

    diopiDtype_t tmp0Dtype;
    diopiGetTensorDtype(tmp0, &tmp0Dtype);

    // convert memoryformat
    bool needConvertMemoryFormat = true;
    if(targetMemoryFormats.size() == 0) {
        needConvertMemoryFormat = false;
    }
    for(auto memoryFormat : targetMemoryFormats) {
        if(isContiguous(srcSize, srcStride, memoryFormat)){
            needConvertMemoryFormat = false;
            break;
        }
    }
    diopiSize_t dstStride = srcStride;
    diopiSize_t dstSize = srcSize;
    if (needConvertMemoryFormat){
        diopiContiguous(ctx, &memoryFormatedTensor, tmp0, targetMemoryFormats[0]);
        convertType.setMemoryFormatConverted();
        diopiGetTensorStride(memoryFormatedTensor, &dstStride);
        diopiGetTensorShape(memoryFormatedTensor, &dstSize);
    }else{
        memoryFormatedTensor = tmp0;
    }

    *dst = memoryFormatedTensor;
    return convertType;
}



template<class T, class strategy = NoCast>
ConvertType requireTensorIfMemoryFormatConvert(diopiContextHandle_t ctx, T src, T* dst,
                     std::vector<diopiMemoryFormat_t> supportMemoryFormats = defaultFormats) {
    ConvertType convertType;
    if (!src) {
        *dst = src;
        return convertType;
    }
    diopiDtype_t srcDtype, dstDtype;
    diopiGetTensorDtype(src, &srcDtype);
    diopiSize_t srcSize, srcStride;
    diopiGetTensorShape(src, &srcSize);
    diopiGetTensorStride(src, &srcStride);
    strategy::getDstDtype(srcDtype, dstDtype);
    std::vector<diopiMemoryFormat_t> targetMemoryFormats = obtainTargetMemoryFormats(srcSize.len, supportMemoryFormats);
    diopiTensorHandle_t memoryFormatedTensor = nullptr;
    bool needConvertMemoryFormat = true;
    if(targetMemoryFormats.size() == 0) {
        needConvertMemoryFormat = false;
    }
    for(auto memoryFormat : targetMemoryFormats) {
        if(isContiguous(srcSize, srcStride, memoryFormat)){
            needConvertMemoryFormat = false;
            break;
        }
    }
    diopiSize_t dstStride = srcStride;
    std::vector<int64_t> dstStrideVec;
    diopiSize_t dstSize = srcSize;
    if (needConvertMemoryFormat){
        diopiMemoryFormat_t targetMemoryForamt = targetMemoryFormats[0];
        dstStrideVec = calcStrides(dstSize, targetMemoryForamt);
        dstStride.data = dstStrideVec.data();
        dstStride.len= dstStrideVec.size();
        convertType.setMemoryFormatConverted();
    }

    // convert Dtype
    if(srcDtype != dstDtype){
        convertType.setDtypeConverted();
    }
    if(convertType.isConverted()){
        diopiDevice_t device;
        diopiGetTensorDevice(src, &device);
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &dstSize, &dstStride, dstDtype, device);
        *dst = tmp;
    }
    else{
        *dst = src;
    }

    return convertType;
}

template <typename Adaptor, typename... Args>
void dispatch_diopi(diopiContextHandle_t ctx, Args&&... args) {
    auto adaptor = Adaptor();
    adaptor(ctx, std::forward<Args>(args)...);
}

inline bool isEqualDiopiSize(diopiSize_t val1, diopiSize_t val2) {
    if (val1.len == val2.len) {
        for(int i = 0; i < val1.len; ++i) {
            if (val1.data[i] != val2.data[i]){
                return false;
            }
        }
        return true;
    }
    return false;
}

const char* getDiopiErrorStr(diopiError_t err) {
    switch (err) {
        case diopiErrorOccurred:
            return "diopiErrorOccurred";
        case diopiNotInited:
            return "diopiNotInited";
        case diopiNoRegisteredStreamCreateFunction:
            return "diopiNoRegisteredStreamCreateFunction";
        case diopiNoRegisteredStreamDestoryFunction:
            return "diopiNoRegisteredStreamDestoryFunction";
        case diopiNoRegisteredStreamSyncFunction:
            return "diopiNoRegisteredStreamSyncFunction";
        case diopiNoRegisteredDeviceMemoryMallocFunction:
            return "diopiNoRegisteredDeviceMemoryMallocFunction";
        case diopiNoRegisteredDeviceMemoryFreeFunction:
            return "diopiNoRegisteredDeviceMemoryFreeFunction";
        case diopiNoRegisteredDevice2DdeviceMemoryCopyFunction:
            return "diopiNoRegisteredDevice2DdeviceMemoryCopyFunction";
        case diopiNoRegisteredDevice2HostMemoryCopyFunction:
            return "diopiNoRegisteredDevice2HostMemoryCopyFunction";
        case diopiNoRegisteredHost2DeviceMemoryCopyFunction:
            return "diopiNoRegisteredHost2DeviceMemoryCopyFunction";
        case diopiNoRegisteredGetLastErrorFunction:
            return "diopiNoRegisteredGetLastErrorFunction";
        case diopi5DNotSupported:
            return "diopi5DNotSupported";
        case diopiDtypeNotSupported:
            return "diopiDtypeNotSupported";
        default:
            return "diopiUnexpectedError";
    }
}

template<class strategy = NoCast>
class DiopiTensorWrapper {

    // forbid copy/move constructor/assignment
    DiopiTensorWrapper(const DiopiTensorWrapper&) = delete;
    DiopiTensorWrapper& operator=(const DiopiTensorWrapper&) = delete;
    DiopiTensorWrapper(DiopiTensorWrapper&&) = delete;
    DiopiTensorWrapper& operator=(DiopiTensorWrapper&&) = delete;

private:
    diopiContextHandle_t ctx_;
    diopiTensorHandle_t payload_;
    diopiTensorHandle_t tmp_ = nullptr;
    ConvertType convertType_;

public:
    DiopiTensorWrapper(diopiContextHandle_t ctx, diopiTensorHandle_t payload,
                       std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats, bool inp = false)
                       : ctx_(ctx)
                       , payload_(payload) {
            TimeElapsed castOutConstructTimeElapsed("out_construct");
            if (inp){
                convertType_ = castImpl<diopiTensorHandle_t, strategy>(ctx, payload_, &tmp_, supportMemoryFormat);
            }
            else {
                convertType_ = requireTensorIfMemoryFormatConvert<diopiTensorHandle_t, strategy>(ctx, payload_, &tmp_, supportMemoryFormat);
            }
    }

    ~DiopiTensorWrapper() {
        TimeElapsed castOutDeconstructTimeElapsed("out_deconstruct");
        if (!convertType_.isConverted()) {
            if (tmp_) {
                payload_ = tmp_;
            }
            return;
        }
        diopiTensorHandle_t memoryFormatedTensor = tmp_;
        if (convertType_.isMemoryFormatConverted()){
            diopiContiguous(ctx_, &memoryFormatedTensor, tmp_, diopiMemoryFormat_t::Contiguous);
            diopiCopyInp(ctx_, memoryFormatedTensor, payload_);
        }
        if (convertType_.isDtypeConverted()){
            diopiCastDtype(ctx_, payload_, memoryFormatedTensor);
        }

       // if (convertType_.isDtypeConverted() && !convertType_.isMemoryFormatConverted()) {
       //     diopiCastDtype(ctx_, payload_, tmp_);
       // } else if (!convertType_.isDtypeConverted() && convertType_.isMemoryFormatConverted()) {
       //     diopiCopyInp(ctx_, tmp_, payload_);
       // } else {
       //     diopiDtype_t dtype;
       //     diopiGetTensorDtype(tmp_, &dtype);
       //     diopiSize_t size, stride, dstStride;
       //     diopiGetTensorShape(payload_, &size);
       //     diopiGetTensorStride(payload_, &stride);
       //     diopiDevice_t device;
       //     diopiGetTensorDevice(payload_, &device);
       //     diopiTensorHandle_t tmp = nullptr;
       //     diopiRequireTensor(ctx_, &tmp, &size, &stride, dtype, device);
       //     diopiCopyInp(ctx_, tmp_, tmp);
       //     diopiCastDtype(ctx_, payload_, tmp);
       // }
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
    TimeElapsed adaptorTimeElapsed("${op_name}_adaptor");
    ${new_input}
    {
        TimeElapsed castInputTimeElapsed("${op_name}_cast_input");
        ${cast_input}
    }

    ${cast_output}
    diopiError_t ret;
    {
        TimeElapsed opTimeElapsed("${op_name}");
        ret = ::${call_func}
    }
    return ret;
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