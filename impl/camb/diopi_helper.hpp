/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CAMB_DIOPI_HELPER_HPP_
#define IMPL_CAMB_DIOPI_HELPER_HPP_

#include <cnnl.h>
#include <cnrt.h>
#include <diopi/diopirt.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "error.hpp"
#include "impl_functions.hpp"

#define DIOPI_CHECK(cond, fmt, args...)                                                      \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            impl::camb::setLastErrorString(#fmt " at %s:%d.\n", ##args, __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                         \
            return diopiErrorOccurred;                                                       \
        }                                                                                    \
    } while (false);

inline void debugPrintBacktrace() {
#ifdef DEBUG_MODE
    impl::camb::printBacktrace();
#endif
}

#define DIOPI_CHECK_NULLPTR_ABORT(variable)                                                      \
    do {                                                                                         \
        if (variable == nullptr) {                                                               \
            printf("The variable `" #variable "` is not defined at %s:%d ", __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                             \
            debugPrintBacktrace();                                                               \
            abort();                                                                             \
        }                                                                                        \
    } while (false);

#define DIOPI_CHECK_ABORT(cond, fmt, args...)                        \
    do {                                                             \
        if (!(cond)) {                                               \
            printf(fmt " at %s:%d ", ##args, __FILE__, __LINE__);    \
            printf("%s", impl::camb::cambGetLastErrorString(false)); \
            debugPrintBacktrace();                                   \
            abort();                                                 \
        }                                                            \
    } while (false);

#define DIOPI_CALL(Expr)                                                                                                            \
    do {                                                                                                                            \
        diopiError_t ret = Expr;                                                                                                    \
        if (diopiSuccess != ret) {                                                                                                  \
            impl::camb::setLastErrorString("%s: %s at %s:%d\n", ::impl::camb::getDiopiErrorStr(ret), __func__, __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                                                                \
            return ret;                                                                                                             \
        }                                                                                                                           \
    } while (false);

namespace impl {
namespace camb {

using MemoryFormat = diopiMemoryFormat_t;

class DiopiDataType final {
public:
    static bool isInteger(diopiDtype_t dtype) { return dtype < 8; }
    static bool isFloatPoint(diopiDtype_t dtype) { return (dtype <= 10 && dtype >= 8) || dtype == 12 || dtype == 13; }
    static diopiDtype_t complexDtype2Real(diopiDtype_t complexDtype) {
        switch (complexDtype) {
            case diopi_dtype_complex128:
                return diopi_dtype_float64;
            case diopi_dtype_complex64:
                return diopi_dtype_float32;
            case diopi_dtype_complex32:
                return diopi_dtype_float16;
            default:
                setLastErrorString("Unsupported ComplexDatatype %s at %s:%d", DiopiDataType::dataTypeStr(complexDtype), __FILE__, __LINE__);
                return diopi_dtype_unsupported;
        }
    }
    static diopiDtype_t realDtype2Complex(diopiDtype_t realDtype) {
        switch (realDtype) {
            case diopi_dtype_float64:
                return diopi_dtype_complex128;
            case diopi_dtype_float32:
                return diopi_dtype_float32;
            case diopi_dtype_float16:
                return diopi_dtype_complex32;
            default:
                setLastErrorString("Unsupported ComplexDatatype %s at %s:%d", DiopiDataType::dataTypeStr(realDtype), __FILE__, __LINE__);
                return diopi_dtype_unsupported;
        }
    }
    static const char* dataTypeStr(diopiDtype_t dtype) {
        switch (dtype) {
            case diopi_dtype_int8:
                return "diopi_dtype_int8";
            case diopi_dtype_uint8:
                return "diopi_dtype_uint8";
            case diopi_dtype_int16:
                return "diopi_dtype_int16";
            case diopi_dtype_uint16:
                return "diopi_dtype_uint16";
            case diopi_dtype_int32:
                return "diopi_dtype_int32";
            case diopi_dtype_uint32:
                return "diopi_dtype_uint32";
            case diopi_dtype_int64:
                return "diopi_dtype_int64";
            case diopi_dtype_uint64:
                return "diopi_dtype_uint64";
            case diopi_dtype_float16:
                return "diopi_dtype_float16";
            case diopi_dtype_float32:
                return "diopi_dtype_float32";
            case diopi_dtype_float64:
                return "diopi_dtype_float64";
            case diopi_dtype_bool:
                return "diopi_dtype_bool";
            case diopi_dtype_bfloat16:
                return "diopi_dtype_bfloat16";
            case diopi_dtype_tfloat32:
                return "diopi_dtype_tfloat32";
            case diopi_dtype_complex32:
                return "diopi_dtype_complex32";
            case diopi_dtype_complex64:
                return "diopi_dtype_complex64";
            case diopi_dtype_complex128:
                return "diopi_dtype_complex128";
            default:
                setLastErrorString("dtype:%d is not support at %s:%d.\n", dtype, __FILE__, __LINE__);
        }
        return "";
    }
};

template <typename T>
diopiScalar_t constructDiopiScalarT(diopiDtype_t dtype, T val) {
    diopiScalar_t scalar;
    scalar.stype = dtype;
    if (DiopiDataType::isFloatPoint(dtype)) {
        scalar.fval = static_cast<double>(val);
    } else {
        scalar.ival = static_cast<int64_t>(val);
    }
    return scalar;
}

class DiopiTensor final {
public:
    DiopiTensor() = default;

    // default shallow copy/assignment, it will not change the address of tensor_
    DiopiTensor(const DiopiTensor&) = default;
    DiopiTensor& operator=(const DiopiTensor&) = default;

    explicit DiopiTensor(const diopiTensorHandle_t& tensor);

    explicit DiopiTensor(const diopiConstTensorHandle_t& tensor) : DiopiTensor(const_cast<diopiTensorHandle_t>(tensor)) {}

    explicit operator diopiTensorHandle_t() { return tensor_; }

    diopiDevice_t device() const;

    diopiDtype_t dtype() const;

    DiopiTensor& setDtype(diopiDtype_t dtype) {
        dtype_ = dtype;
        return *this;
    }

    DiopiTensor viewAsComplex() const;

    DiopiTensor viewAsReal() const;

    const std::vector<int64_t>& shape() const;

    int64_t size(int i) const {
        if (i < 0) {
            i = shape_.size() + i;
        }
        return shape()[i];
    }

    const std::vector<int64_t>& stride() const;

    int64_t numel() const;
    int64_t elemsize() const;
    int64_t dim() const { return static_cast<int64_t>(this->shape().size()); }

    /* DEPRECATED AND WILL BE REMOVED */
    DiopiTensor contiguous(diopiContextHandle_t ctx, MemoryFormat format = MemoryFormat::Contiguous);

    bool isContiguous(MemoryFormat format = MemoryFormat::Contiguous) const;

    DiopiTensor& asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride);

    DiopiTensor& unsqueeze(int dim);

    bool defined() const { return tensor_ != nullptr; }

    DiopiTensor& view(const std::vector<int64_t> shape);

    void* data();
    const void* data() const;

    MemoryFormat suggestMemoryFormat();

    diopiTensorHandle_t tensorHandle();
    diopiConstTensorHandle_t tensorHandle() const;

    bool isSame(DiopiTensor t) { return this->tensorHandle() == t.tensorHandle(); }

protected:
    diopiTensorHandle_t tensor_ = nullptr;
    diopiDtype_t dtype_{diopi_dtype_float32};
    std::vector<int64_t> shape_{0};
    std::vector<int64_t> stride_{0};
};

DiopiTensor makeTensor(diopiContextHandle_t ctx, const diopiScalar_t* pScalar);

DiopiTensor ones(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype, MemoryFormat memoryFormat);

inline DiopiTensor requiresBuffer(diopiContextHandle_t ctx, int64_t numBytes) {
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireBuffer(ctx, &tensor, numBytes, diopi_device);
    return DiopiTensor(tensor);
}

inline cnrtQueue_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t streamHandle;
    diopiGetStream(ctx, &streamHandle);
    return static_cast<cnrtQueue_t>(streamHandle);
}

template <typename T>
std::vector<T> diopiSizeT2Vector(diopiSize_t size) {
    return std::vector<T>(size.data, size.data + size.len);
}

inline diopiSize_t vec2diopiSizeT(const std::vector<int64_t>& sizeIn) {
    diopiSize_t diopiSize{sizeIn.data(), static_cast<int64_t>(sizeIn.size())};
    return diopiSize;
}

inline void syncStreamInCtx(diopiContextHandle_t ctx) {
    cnrtQueue_t queue = getStream(ctx);
    cnrtQueueSync(queue);
    return;
}

inline const char* reductionStr(diopiReduction_t reduction) {
    switch (reduction) {
        case ReductionNone:
            return "ReductionNone";
        case ReductionSum:
            return "ReductionSum";
        case ReductionMean:
            return "ReductionMean";
        default:
            return "not supported reduction method";
    }
}

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_DIOPI_HELPER_HPP_
