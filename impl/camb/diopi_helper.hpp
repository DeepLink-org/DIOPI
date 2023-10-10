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
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "error.hpp"
#include "impl_functions.hpp"

namespace impl {
namespace camb {

void getFuncName(const char* expr, char* name);

}  // namespace camb
}  // namespace impl

#define DIOPI_CHECK(cond, fmt, args...)                                                      \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            impl::camb::setLastErrorString(#fmt " at %s:%d.\n", ##args, __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                         \
            return diopiErrorOccurred;                                                       \
        }                                                                                    \
    } while (false);

#define DIOPI_CHECK_NULLPTR_ABORT(variable)                                                      \
    do {                                                                                         \
        if (variable == nullptr) {                                                               \
            printf("The variable `" #variable "` is not defined at %s:%d ", __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                             \
            abort();                                                                             \
        }                                                                                        \
    } while (false);

#define DIOPI_CHECK_ABORT(cond, fmt, args...)                        \
    do {                                                             \
        if (!(cond)) {                                               \
            printf(#fmt " at %s:%d ", ##args, __FILE__, __LINE__);   \
            printf("%s", impl::camb::cambGetLastErrorString(false)); \
            abort();                                                 \
        }                                                            \
    } while (false);

#define DIOPI_RECORD_START(Expr)              \
    const int kFuncNameMaxLen = 100;          \
    char funcName[kFuncNameMaxLen];           \
    impl::camb::getFuncName(#Expr, funcName); \
    diopiRecordStart(funcName, &record);

// cant' use this macro DIOPI_RECORD_END alone, but use it in pairs with DIOPI_RECORD_START
#define DIOPI_RECORD_END diopiRecordEnd(&record);

extern bool isRecordOn;

#define DIOPI_CALL(Expr)                                                                                                            \
    do {                                                                                                                            \
        void* record = nullptr;                                                                                                     \
        if (isRecordOn) {                                                                                                           \
            DIOPI_RECORD_START(Expr);                                                                                               \
        }                                                                                                                           \
        diopiError_t ret = Expr;                                                                                                    \
        if (isRecordOn) {                                                                                                           \
            DIOPI_RECORD_END;                                                                                                       \
        }                                                                                                                           \
        if (diopiSuccess != ret) {                                                                                                  \
            impl::camb::setLastErrorString("%s: %s at %s:%d\n", ::impl::camb::getDiopiErrorStr(ret), __func__, __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                                                                \
            return ret;                                                                                                             \
        }                                                                                                                           \
    } while (false);

namespace impl {
namespace camb {

class DiopiDataType final {
public:
    static bool isInteger(diopiDtype_t dtype);
    static bool isFloatPoint(diopiDtype_t dtype);
    static diopiDtype_t complexDtype2Real(diopiDtype_t complexDtype);
    static diopiDtype_t realDtype2Complex(diopiDtype_t realDtype);
    static const char* dataTypeStr(diopiDtype_t dtype);
};

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

    template <typename T>
    std::vector<T> shape() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return std::vector<T>(shape_.begin(), shape_.end());
    }

    const std::vector<int64_t>& shape() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return shape_;
    }

    const std::vector<int64_t>& stride() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return stride_;
    }

    int64_t size(int i) const {
        if (i < 0) {
            i = shape_.size() + i;
        }
        return shape()[i];
    }

    int64_t numel() const;
    int64_t elemsize() const;
    int64_t dim() const { return static_cast<int64_t>(this->shape().size()); }

    bool isContiguous(diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) const;

    DiopiTensor& asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride);

    DiopiTensor& unsqueeze(int dim);

    bool defined() const { return tensor_ != nullptr; }

    DiopiTensor& view(const std::vector<int64_t> shape);

    void* data();
    const void* data() const;

    diopiTensorHandle_t tensorHandle() { return tensor_; }
    diopiConstTensorHandle_t tensorHandle() const { return tensor_; }

    bool isSame(DiopiTensor t) { return this->tensorHandle() == t.tensorHandle(); }

protected:
    diopiTensorHandle_t tensor_ = nullptr;
    diopiDtype_t dtype_{diopi_dtype_unsupported};
    std::vector<int64_t> shape_{0};
    std::vector<int64_t> stride_{0};
};

DiopiTensor makeTensor(diopiContextHandle_t ctx, const diopiScalar_t* pScalar);

DiopiTensor ones(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype);

DiopiTensor zeros(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype);

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype, diopiMemoryFormat_t memoryFormat);

DiopiTensor requiresBuffer(diopiContextHandle_t ctx, int64_t numBytes);

cnrtQueue_t getStream(diopiContextHandle_t ctx);

template <typename T>
std::vector<T> diopiSizeT2Vector(diopiSize_t size) {
    return std::vector<T>(size.data, size.data + size.len);
}

diopiSize_t vec2diopiSizeT(const std::vector<int64_t>& sizeIn);

void syncStreamInCtx(diopiContextHandle_t ctx);

const char* reductionStr(diopiReduction_t reduction);

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

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_DIOPI_HELPER_HPP_
