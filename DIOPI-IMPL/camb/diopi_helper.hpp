/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef IMPL_CAMB_DIOPI_HELPER_HPP_
#define IMPL_CAMB_DIOPI_HELPER_HPP_

#include <cnnl.h>
#include <cnrt.h>
#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <cstdio>
#include <iostream>
#include <utility>
#include <vector>

#include "error.hpp"

namespace impl {
namespace camb {

#define DIOPI_CHECK(cond, str)                                             \
    do {                                                                   \
        if (!(cond)) {                                                     \
            set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                     \
        }                                                                  \
    } while (false);

#define DIOPI_CHECK_NULLPTR_ABORT(variable)                                                      \
    do {                                                                                         \
        if (variable == nullptr) {                                                               \
            printf("The variable `" #variable "` is not defined at %s:%d ", __FILE__, __LINE__); \
            abort();                                                                             \
        }                                                                                        \
    } while (false);

#define DIOPI_CHECK_ABORT(cond, fmt, args...)                    \
    do {                                                         \
        if (!(cond)) {                                           \
            printf(#fmt " at %s:%d ", args, __FILE__, __LINE__); \
            abort();                                             \
        }                                                        \
    } while (false);

#define DIOPI_CALL(Expr)           \
    do {                           \
        diopiError_t ret = Expr;   \
        if (diopiSuccess != ret) { \
            return ret;            \
        }                          \
    } while (false);

enum class MemoryFormat : size_t { Contiguous = 0, ChannelsLast = 1, ChannelsLast3d = 2, Preserve = 3 };

class DiopiDataType final {
public:
    static bool isInteger(diopiDtype_t dtype) { return dtype < 8; }
    static bool isFloatPoint(diopiDtype_t dtype) { return dtype <= 10 && dtype >= 8 || dtype == 12 || dtype == 13; }
};

class DiopiTensor final {
public:
    DiopiTensor() = default;
    explicit DiopiTensor(const diopiTensorHandle_t& tensor) : tensor_(tensor) {
        if (tensor_ != nullptr) {
            diopiSize_t diopiShape;
            diopiSize_t diopiStride;
            diopiGetTensorShape(tensor_, &diopiShape);
            std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
            diopiGetTensorStride(tensor_, &diopiStride);
            std::vector<int64_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
            shape_ = std::move(shapeTmp);
            stride_ = std::move(strideTmp);
        }
    }
    explicit DiopiTensor(const diopiConstTensorHandle_t& tensor) : DiopiTensor(const_cast<diopiTensorHandle_t>(tensor)) {}

    explicit operator diopiTensorHandle_t() { return tensor_; }

    diopiDevice_t device() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        diopiDevice_t device;
        diopiGetTensorDevice(tensor_, &device);
        return device;
    }
    diopiDtype_t dtype() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        diopiDtype_t dtype;
        diopiGetTensorDtype(tensor_, &dtype);
        return dtype;
    }

    const std::vector<int64_t>& shape() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return shape_;
    }
    const std::vector<int64_t>& stride() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return stride_;
    }

    int64_t numel() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        int64_t numel;
        diopiGetTensorNumel(tensor_, &numel);
        return numel;
    }
    int64_t elemsize() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        int64_t elemsize;
        diopiGetTensorElemSize(tensor_, &elemsize);
        return elemsize;
    }
    int64_t dim() const { return this->shape().size(); }

    DiopiTensor contiguous(diopiContextHandle_t ctx, MemoryFormat format = MemoryFormat::Contiguous) {
        /* Returns a new Tensor in new memory format, without data copy */
        if (this->is_contiguous(format)) return *this;
        MemoryFormat format_self;
        int64_t dim = this->dim();
        std::vector<int64_t> strides(dim);
        int64_t stride = 1;
        if (format == MemoryFormat::Contiguous) {
            for (size_t i = dim; i > 0; --i) {
                strides[i - 1] = stride;
                if (shape_[i - 1] == 0) continue;
                if (shape_[i - 1] == -1) stride = -1;
                if (stride != -1) stride *= shape_[i - 1];
            }
        } else if (format == MemoryFormat::ChannelsLast) {
            for (auto k : {1, 3, 2, 0}) {
                strides[k] = stride;
                if (shape_[k] == 0) continue;
                if (shape_[k] == -1) stride = -1;
                if (stride != -1) stride *= shape_[k];
            }
        }
        diopiSize_t stride_diopi(strides.data(), static_cast<int64_t>(strides.size()));
        diopiSize_t shape_diopi(this->shape().data(), this->shape().size());
        diopiTensorHandle_t tensor;
        diopiRequireTensor(ctx, &tensor, &shape_diopi, &stride_diopi, this->dtype(), this->device());
        return DiopiTensor(tensor);
    }

    bool is_contiguous(MemoryFormat format = MemoryFormat::Contiguous) {
        int64_t stride = 1;
        int64_t dim = this->dim();
        auto strides = this->stride();
        auto shape = this->shape();

        if (format == MemoryFormat::Contiguous) {
            for (int i = dim - 1; i >= 0; i--) {
                if (strides[i] != stride) {
                    return false;
                }
                stride *= shape[i];
            }
        } else if (format == MemoryFormat::ChannelsLast) {
            if (strides.size() != 4) return false;
            for (auto i : {1, 3, 2, 0}) {
                if (strides[i] != stride) {
                    return false;
                }
                stride *= shape[i];
            }
        }
        return true;
    }

    bool defined() const {
        if (tensor_ == nullptr) return false;
        return this->numel() != 0;
    }

    void reshape(const std::vector<int64_t> shape) { this->shape_ = shape; }

    void* data() {
        void* p = nullptr;
        diopiGetTensorData(tensor_, &p);
        return p;
    }
    const void* data() const {
        const void* p = nullptr;
        diopiGetTensorDataConst(tensor_, &p);
        return p;
    }

    diopiTensorHandle_t tensor_handle() {
        return tensor_;
    }

    diopiConstTensorHandle_t tensor_handle() const{
        return tensor_;
    }

protected:
    diopiTensorHandle_t tensor_ = 0;
    std::vector<int64_t> shape_{0};
    std::vector<int64_t> stride_{0};
};

inline auto makeTensor(diopiContextHandle_t ctx, const diopiScalar_t* pScalar) -> DiopiTensor {
    diopiTensorHandle_t tensor;
    std::vector<int64_t> shape{1};
    diopiSize_t size(shape.data(), 1);
    diopiRequireTensor(ctx, &tensor, &size, nullptr, pScalar->stype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor ones(diopiContextHandle_t ctx, std::vector<int64_t> size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor;
    diopiSize_t size_(size.data(), size.size());
    diopiRequireTensor(ctx, &tensor, &size_, nullptr, dtype, diopi_device);
    diopiScalar_t scalar = {dtype, 1.0};
    if (DiopiDataType().isInteger(dtype)) scalar = {dtype, 1};
    diopiFill(ctx, tensor, &scalar);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, diopiDtype_t dtype) {
    diopiSize_t size_(size.data(), size.size());
    diopiSize_t stride_(stride.data(), stride.size());
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size_, &stride_, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiSize_t size_(size.data(), size.size());
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size_, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresBuffer(diopiContextHandle_t ctx, int64_t num_bytes) {
    diopiTensorHandle_t tensor;
    diopiRequireBuffer(ctx, &tensor, num_bytes, diopi_device);
    return DiopiTensor(tensor);
}

inline cnrtQueue_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    return static_cast<cnrtQueue_t>(stream_handle);
}

template <typename T>
inline std::vector<T> diopiSize_t2Vector(diopiSize_t size, T) {
    return std::vector<T>(size.data(), size.data() + size.len);
}

inline diopiSize_t vec2diopiSize_t(const std::vector<int64_t>& sizeIn) {
    diopiSize_t diopiSize(sizeIn.data(), sizeIn.size());
    return diopiSize;
}
}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_DIOPI_HELPER_HPP_
