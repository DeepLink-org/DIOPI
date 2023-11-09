#ifndef DIOPI_ADAPTOR_CSRC_CONVERT_HPP_
#define DIOPI_ADAPTOR_CSRC_CONVERT_HPP_

#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

std::vector<int64_t> calcStrides(diopiSize_t size, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

bool isLikeChannelsLast(diopiConstTensorHandle_t tensor, bool checkContiguous, diopiMemoryFormat_t format = diopiMemoryFormat_t::ChannelsLast);

inline diopiMemoryFormat_t probableMemoryFormat(diopiConstTensorHandle_t tensor, bool exactMatch = false);

bool isContiguous(diopiSize_t size, diopiSize_t strideDiopi, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

std::vector<diopiMemoryFormat_t> matchMemoryFormatBySize(size_t sizeLen);

std::vector<diopiMemoryFormat_t> setIntersection(std::vector<diopiMemoryFormat_t> matchedMemoryFormat, std::vector<diopiMemoryFormat_t> supportMemoryFormat);

std::vector<diopiMemoryFormat_t> obtainTargetMemoryFormats(size_t shapeLen, std::vector<diopiMemoryFormat_t> supportMemoryFormats);

class TimeElapsedRecord {
public:
    TimeElapsedRecord(const char *fileName) : enableTiming_(false) {
        const char *enableEnvVar = getenv("DIOPI_ENABLE_TIMING");
        if (enableEnvVar && strcmp(enableEnvVar, "OFF") != 0 && strcmp(enableEnvVar, "0") != 0) {
            enableTiming_ = true;
            stream_ = std::ofstream(fileName, std::ios::out | std::ios::trunc);
        }
    }
    bool isEnableTiming() { return enableTiming_; }
    std::ofstream &getOStream() { return stream_; }

private:
    bool enableTiming_;
    std::ofstream stream_;
};

class TimeElapsed {
public:
    TimeElapsed(const char *opName) : opName_(opName) {
        if (timeElapsedRecord.isEnableTiming()) {
            start_ = std::chrono::steady_clock::now();
        }
    }
    ~TimeElapsed() {
        if (timeElapsedRecord.isEnableTiming()) {
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start_;  // ms
            double elapsedTime = elapsed.count();
            timeElapsedRecord.getOStream() << opName_ << ": " << elapsedTime << "ms" << std::endl;
        }
    }

private:
    const char *opName_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
    static TimeElapsedRecord timeElapsedRecord;
};

class ConvertType {
public:
    ConvertType() : val_(0){};
    void setDtypeConverted() { val_ |= 0x1u; }
    void setMemoryFormatConverted() { val_ |= 0x2u; }
    bool isDtypeConverted() { return val_ & 0x1u; }
    bool isMemoryFormatConverted() { return val_ & 0x2u; }
    bool isConverted() { return static_cast<bool>(val_); }

private:
    unsigned char val_;
};

template <typename T>
struct RemoveConst {
    using type = T;
};

template <>
struct RemoveConst<diopiConstTensorHandle_t> {
    using type = diopiTensorHandle_t;
};

template <class T, class strategy>
ConvertType castImpl(diopiContextHandle_t ctx, T src, T *dst, std::vector<diopiMemoryFormat_t> supportMemoryFormats) {
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
    if (needConvertDtype) {
        diopiRequireTensor(ctx, &tmp0, &srcSize, &srcStride, dstDtype, device);
        diopiCastDtype(ctx, tmp0, src);
        convertType.setDtypeConverted();
    } else {
        tmp0 = const_cast<typename RemoveConst<T>::type>(src);
    }

    diopiDtype_t tmp0Dtype;
    diopiGetTensorDtype(tmp0, &tmp0Dtype);

    // convert memoryformat
    bool needConvertMemoryFormat = true;
    if (targetMemoryFormats.empty()) {
        needConvertMemoryFormat = false;
    }
    for (auto memoryFormat : targetMemoryFormats) {
        if (isContiguous(srcSize, srcStride, memoryFormat)) {
            needConvertMemoryFormat = false;
            break;
        }
    }
    diopiSize_t dstStride = srcStride;
    diopiSize_t dstSize = srcSize;
    if (needConvertMemoryFormat) {
        diopiContiguous(ctx, &memoryFormatedTensor, tmp0, targetMemoryFormats[0]);
        convertType.setMemoryFormatConverted();
        diopiGetTensorStride(memoryFormatedTensor, &dstStride);
        diopiGetTensorShape(memoryFormatedTensor, &dstSize);
    } else {
        memoryFormatedTensor = tmp0;
    }

    *dst = memoryFormatedTensor;
    return convertType;
}

template <class T, class strategy>
ConvertType requireTensorIfMemoryFormatConvert(diopiContextHandle_t ctx, T src, T *dst, std::vector<diopiMemoryFormat_t> supportMemoryFormats) {
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
    bool needConvertMemoryFormat = true;
    if (targetMemoryFormats.empty()) {
        needConvertMemoryFormat = false;
    }
    for (auto memoryFormat : targetMemoryFormats) {
        if (isContiguous(srcSize, srcStride, memoryFormat)) {
            needConvertMemoryFormat = false;
            break;
        }
    }
    diopiSize_t dstStride = srcStride;
    std::vector<int64_t> dstStrideVec;
    diopiSize_t dstSize = srcSize;
    if (needConvertMemoryFormat) {
        diopiMemoryFormat_t targetMemoryForamt = targetMemoryFormats[0];
        dstStrideVec = calcStrides(dstSize, targetMemoryForamt);
        dstStride.data = dstStrideVec.data();
        dstStride.len = dstStrideVec.size();
        convertType.setMemoryFormatConverted();
    }

    // convert Dtype
    if (srcDtype != dstDtype) {
        convertType.setDtypeConverted();
    }
    if (convertType.isConverted()) {
        diopiDevice_t device;
        diopiGetTensorDevice(src, &device);
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &dstSize, &dstStride, dstDtype, device);
        *dst = tmp;
    } else {
        *dst = src;
    }

    return convertType;
}

template <typename Adaptor, typename... Args>
void dispatchDiopi(diopiContextHandle_t ctx, Args &&...args) {
    auto adaptor = Adaptor();
    adaptor(ctx, std::forward<Args>(args)...);
}

inline bool isEqualDiopiSize(diopiSize_t val1, diopiSize_t val2) {
    if (val1.len == val2.len) {
        for (int i = 0; i < val1.len; ++i) {
            if (val1.data[i] != val2.data[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

template <class strategy>
class DiopiTensorWrapper {
public:
    // forbid copy/move constructor/assignment
    DiopiTensorWrapper(const DiopiTensorWrapper &) = delete;
    DiopiTensorWrapper &operator=(const DiopiTensorWrapper &) = delete;
    DiopiTensorWrapper(DiopiTensorWrapper &&) = delete;
    DiopiTensorWrapper &operator=(DiopiTensorWrapper &&) = delete;

private:
    diopiContextHandle_t ctx_;
    diopiTensorHandle_t payload_;
    diopiTensorHandle_t tmp_ = nullptr;
    ConvertType convertType_;

public:
    DiopiTensorWrapper(diopiContextHandle_t ctx, diopiTensorHandle_t payload, std::vector<diopiMemoryFormat_t> supportMemoryFormat, bool inp = false)
        : ctx_(ctx), payload_(payload) {
        TimeElapsed castOutConstructTimeElapsed("out_construct");
        if (inp) {
            convertType_ = castImpl<diopiTensorHandle_t, strategy>(ctx, payload_, &tmp_, supportMemoryFormat);
        } else {
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
        if (convertType_.isMemoryFormatConverted()) {
            diopiCopyInp(ctx_, memoryFormatedTensor, payload_);
        }
        if (convertType_.isDtypeConverted()) {
            diopiCastDtype(ctx_, payload_, memoryFormatedTensor);
        }

        // if (convertType_.isDtypeConverted() &&
        // !convertType_.isMemoryFormatConverted()) {
        //     diopiCastDtype(ctx_, payload_, tmp_);
        // } else if (!convertType_.isDtypeConverted() &&
        // convertType_.isMemoryFormatConverted()) {
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
    operator diopiTensorHandle_t() { return tmp_; }
};

#endif  // DIOPI_ADAPTOR_CSRC_CONVERT_HPP_