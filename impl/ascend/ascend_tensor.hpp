#ifndef IMPL_ASCEND_ASCEND_TENSOR_HPP_
#define IMPL_ASCEND_ASCEND_TENSOR_HPP_

#include <acl/acl.h>
#include <diopi/diopirt.h>

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

#define ASCEND_CHECK_ABORT(condition, ...)                            \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
            std::abort();                                             \
        }                                                             \
    } while (0);

#define ASCEND_CHECK_NULLPTR_ABORT(ptr) ASCEND_CHECK_ABORT(ptr, "Variable is nullptr, pls check.")

#define error(...)                               \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");                                \
    std::abort();

#define warning(...)                             \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");

#define info(...)                                \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");

class AscendTensor final {
public:
    explicit AscendTensor(const diopiConstTensorHandle_t& tensor) : tensor_(tensor) {
        if (tensor_ != nullptr) {
            diopiSize_t diopiShape;
            diopiGetTensorShape(tensor_, &diopiShape);
            std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
            shape_ = std::move(shapeTmp);

            diopiSize_t diopiStride;
            diopiGetTensorStride(tensor_, &diopiStride);
            std::vector<int64_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
            stride_ = std::move(strideTmp);
            ASCEND_CHECK_ABORT(stride_.size() == shape_.size(), "stride_.size() == shape_.size() check failed");

            diopiDtype_t diopiDtype;
            diopiGetTensorDtype(tensor_, &diopiDtype);
            dtype_ = diopiDtype;

            diopiDevice_t device;
            diopiGetTensorDevice(tensor_, &device);
            device_ = device;

            diopiGetTensorNumel(tensor_, &numel_);
            diopiGetTensorElemSize(tensor_, &elemsize_);
        }
    }

public:
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

    int64_t dim() const {
        if (shape_.empty()) {
            return 0;
        }
        return static_cast<int64_t>(this->shape().size());
    }

    bool defined() const { return tensor_; }

    int64_t numel() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return numel_;
    }

    int64_t elemsize() const {
        ASCEND_CHECK_NULLPTR_ABORT(tensor_);
        return elemsize_;
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
    aclFormat getAclDataFormat() const;
    aclDataType getAclDataType() const;

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
};

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ASCEND_TENSOR_HPP_
