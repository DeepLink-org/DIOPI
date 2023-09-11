/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_ASCEND_TENSOR_HPP_
#define IMPL_ASCEND_ASCEND_TENSOR_HPP_

#include <diopi/diopirt.h>

#include <vector>

#include "common/debug.hpp"
#include "error.hpp"

namespace impl {
namespace ascend {

class AscendTensor final {
public:
    AscendTensor() = default;
    AscendTensor(const AscendTensor&) = default;
    AscendTensor& operator=(const AscendTensor&) = default;
    explicit AscendTensor(const diopiTensorHandle_t& tensor);
    explicit AscendTensor(const diopiConstTensorHandle_t& tensor) : AscendTensor(const_cast<diopiTensorHandle_t>(tensor)) {}
    // use AscendTensor obj like diopiTensor*
    explicit operator diopiTensorHandle_t() { return tensor_; }
    explicit operator diopiTensorHandle_t() const { return tensor_; }

    diopiDevice_t device() const;
    diopiDtype_t dtype() const;
    AscendTensor& setDtype(diopiDtype_t dtype) {
        dtype_ = dtype;
        return *this;
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

    int64_t numel() const;
    int64_t elemsize() const;
    int64_t dim() const { return static_cast<int64_t>(this->shape().size()); }

    bool isContiguous(diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) const;

    AscendTensor& asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride);

    AscendTensor& unsqueeze(int dim);

    bool defined() const { return tensor_; }

    AscendTensor& view(const std::vector<int64_t> shape);
    void reshape(const std::vector<int64_t>& dims);

    void* data();
    const void* data() const;

    diopiTensorHandle_t tensorHandle();
    diopiConstTensorHandle_t tensorHandle() const;

    bool isSame(AscendTensor t) { return this->tensorHandle() == t.tensorHandle(); }

    int64_t getBaseBufferSize() const;
    std::vector<int64_t> getBaseShape() const;

    inline aclFormat getAclDataFormat() const {
        if (dim() == 4) {
            std::array<int64_t, 4> thStride{stride(0), stride(1), stride(2), stride(3)};
            {
                std::array<int64_t, 4> nchwStride;
                int st = 1;
                for (auto k : {3, 2, 1, 0}) {
                    nchwStride[k] = st;
                    if (shape(k) == 0) continue;
                    if (shape(k) == -1) st = -1;
                    if (st != -1) st *= shape(k);
                }
                if (thStride == nchwStride) {
                    return ACL_FORMAT_NCHW;
                }
            }
            std::array<int64_t, 4> nhwcStride;
            int st = 1;
            for (auto k : {1, 3, 2, 0}) {
                nhwcStride[k] = st;
                if (shape(k) == 0) continue;
                if (shape(k) == -1) st = -1;
                if (st != -1) st *= shape(k);
            }
            if (thStride == nhwcStride) {
                return ACL_FORMAT_NHWC;
            }
            warning("getAclDataFormat error. Acl only support NCHW or NHWC format! but get %s", dumpTensor(tensor_).c_str());
        }
        return ACL_FORMAT_ND;
    }

protected:
    // diopi origin tensor
    diopiTensorHandle_t tensor_ = nullptr;
    diopiDtype_t dtype_{diopi_dtype_unsupported};
    std::vector<int64_t> shape_{0};
    std::vector<int64_t> stride_{0};
};

AscendTensor createAscendTensor(diopiContextHandle_t ctx, const diopiSize_t* size, const diopiSize_t* stride, const diopiDtype_t dtype,
                                const diopiDevice_t dev);
AscendTensor createAscendTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const diopiSize_t* stride, const diopiDtype_t dtype,
                                const diopiDevice_t dev);

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ASCEND_TENSOR_HPP_
