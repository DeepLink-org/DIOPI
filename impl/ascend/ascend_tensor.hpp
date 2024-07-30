#ifndef IMPL_ASCEND_ASCEND_TENSOR_HPP_
#define IMPL_ASCEND_ASCEND_TENSOR_HPP_

#include <acl/acl.h>
#include <diopi/diopirt.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace impl {
namespace ascend {

#define TRACK_ACL(x)                                                    \
    do {                                                                \
        static bool enable = std::getenv("DIOPI_TRACK_ACL") != nullptr; \
        if (enable) {                                                   \
            printf("[%s: %d]:%s\n", __FILE__, __LINE__, x);             \
        }                                                               \
    } while (0);

#define CALL_ACLRT(Expr)                                                                          \
    do {                                                                                          \
        TRACK_ACL(#Expr);                                                                         \
        ::aclError ret = Expr;                                                                    \
        if (ret != ::ACL_SUCCESS) {                                                               \
            throw std::runtime_error(std::string("ascend device error:") + aclGetRecentErrMsg()); \
        }                                                                                         \
    } while (0);

#define CHECK_ASCENDRT(Expr)                                                \
    do {                                                                    \
        TRACK_ACL(#Expr);                                                   \
        ::diopiError_t ret = Expr;                                          \
        if (ret != ::diopiSuccess) {                                        \
            throw std::runtime_error(std::string("call function failed.")); \
        }                                                                   \
    } while (0);

#define ASCEND_CHECK(condition, ...)                                  \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
        }                                                             \
    } while (0);

#define ASCEND_CHECK_ABORT(condition, ...)                            \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
            std::abort();                                             \
        }                                                             \
    } while (0);

#define ASCEND_CHECK_THROW(condition, ...)                                                        \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__);                             \
            printf(__VA_ARGS__);                                                                  \
            printf("\n");                                                                         \
            throw std::runtime_error(std::string("ascend device error:") + aclGetRecentErrMsg()); \
        }                                                                                         \
    } while (0);

#define ASCEND_CHECK_NULLPTR_ABORT(ptr) ASCEND_CHECK_ABORT(ptr, "Variable is nullptr, pls check.")

inline void error(const char* file, int lineNum, const char* funcName, const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("ERROR:[%s:%d in func:%s] : ", file, lineNum, funcName);
    vprintf(format, args);
    printf("\n");
    throw std::runtime_error("error occurs");
}

inline void warning(const char* file, int lineNum, const char* funcName, const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("WARNING:[%s:%d in func:%s]: ", file, lineNum, funcName);
    vprintf(format, args);
    printf("\n");
}

inline void info(const char* file, int lineNum, const char* funcName, const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("INFO:[%s:%d in func:%s]: ", file, lineNum, funcName);
    vprintf(format, args);
    printf("\n");
}

aclFormat inferAclDataFormat(int64_t dim, const int64_t* shape, const int64_t* stride);

constexpr aclDataType diopiDtypeToAclDataType(diopiDtype_t dtype) noexcept {
#define DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype, acl_dtype) \
    case diopi_dtype:                                         \
        return acl_dtype;

    switch (dtype) {
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_bfloat16, ACL_BF16)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_float16, ACL_FLOAT16)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_float32, ACL_FLOAT)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_float64, ACL_DOUBLE)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_int8, ACL_INT8)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_uint8, ACL_UINT8)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_int16, ACL_INT16)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_uint16, ACL_UINT16)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_int32, ACL_INT32)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_uint32, ACL_UINT32)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_int64, ACL_INT64)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_uint64, ACL_UINT64)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_bool, ACL_BOOL)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_complex32, ACL_COMPLEX32)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_complex64, ACL_COMPLEX64)
        DIOPI_DTYPE_TO_ACL_DTYPE_CASE(diopi_dtype_complex128, ACL_COMPLEX128)
        default:
            return ACL_DT_UNDEFINED;
    }
#undef DIOPI_DTYPE_TO_ACL_DTYPE_CASE
}

class AscendTensor final {
public:
    explicit AscendTensor(const diopiConstTensorHandle_t& tensor) : tensor_(tensor) {
        if (tensor_ != nullptr) {
            diopiSize_t diopiShape;
            diopiGetTensorShape(tensor_, &diopiShape);
            shape_.assign(diopiShape.data, diopiShape.data + diopiShape.len);

            diopiSize_t diopiStride;
            diopiGetTensorStride(tensor_, &diopiStride);
            stride_.assign(diopiStride.data, diopiStride.data + diopiStride.len);
            ASCEND_CHECK_ABORT(stride_.size() == shape_.size(), "stride_.size() == shape_.size() check failed.");

            diopiDtype_t diopiDtype;
            diopiGetTensorDtype(tensor_, &diopiDtype);
            dtype_ = diopiDtype;

            diopiDevice_t device;
            diopiGetTensorDevice(tensor_, &device);
            device_ = device;

            diopiGetTensorNumel(tensor_, &numel_);
            diopiGetTensorElemSize(tensor_, &elemsize_);
            diopiGetTensorStorageOffset(tensor_, &storageOffset_);
            diopiGetTensorStorageNbytes(tensor_, &storageNbytes_);
        }
    }

    // default construct
    AscendTensor() = default;
    // Shallow copy.
    AscendTensor(const AscendTensor&) = default;
    AscendTensor& operator=(const AscendTensor&) = default;
    // Use AscendTensor obj like const diopiTensor*
    explicit operator diopiConstTensorHandle_t() { return tensor_; }
    explicit operator diopiConstTensorHandle_t() const { return tensor_; }

    // Get AscendTensor attribute. Those methods can not change ascend tensor attribute.
    diopiDevice_t device() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return device_;
    }

    diopiDtype_t dtype() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return dtype_;
    }

    template <typename T>
    std::vector<T> shape() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return std::vector<T>(shape_.begin(), shape_.end());
    }

    const std::vector<int64_t>& shape() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return shape_;
    }

    const std::vector<int64_t>& stride() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return stride_;
    }

    int64_t shape(int i) const {
        if (i < 0) {
            i = shape_.size() + i;
        }
        return shape()[i];
    }

    int64_t stride(int i) const { return stride()[i]; }

    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }

    bool defined() const { return tensor_; }

    int64_t numel() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return numel_;
    }

    int64_t elemsize() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return elemsize_;
    }

    int64_t storageOffset() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return storageOffset_;
    }

    std::size_t storageNbytes() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return storageNbytes_;
    }

    bool isContiguous(diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) const;

    const void* data() const;

    diopiConstTensorHandle_t tensorHandle() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return tensor_;
    }

    bool isSame(const AscendTensor& t) const { return this->tensorHandle() == t.tensorHandle(); }

    int64_t getAclMemBufferSize() const;
    std::vector<int64_t> getAclMemShape() const;
    aclFormat getAclDataFormat() const { return inferAclDataFormat(dim(), shape_.data(), stride_.data()); }
    aclDataType getAclDataType() const { return diopiDtypeToAclDataType(dtype_); }

    // Those methods may change the class attribute.
    AscendTensor& asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride);
    AscendTensor& unsqueeze(int dim);
    AscendTensor& view(const std::vector<int64_t>& shape);

private:
    // diopi origin tensor
    diopiConstTensorHandle_t tensor_ = nullptr;
    diopiDtype_t dtype_{diopi_dtype_unsupported};
    std::vector<int64_t> shape_{};
    std::vector<int64_t> stride_{};
    diopiDevice_t device_ = diopiDevice_t::diopi_device;
    int64_t numel_{0};
    int64_t elemsize_{0};
    int64_t storageOffset_{0};
    std::size_t storageNbytes_{0};
};

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ASCEND_TENSOR_HPP_
