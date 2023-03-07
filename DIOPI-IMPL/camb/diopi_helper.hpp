/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef IMPL_CAMB_DIOPI_HELPER_HPP_
#define IMPL_CAMB_DIOPI_HELPER_HPP_

#include <diopi/diopirt.h>
#include <cnnl.h>
#include <cnrt.h>
#include <cstdio>

#include <utility>
#include <vector>
#include <iostream>

#include "error.hpp"

namespace impl {
namespace camb {

#define DIOPI_CHECK(cond, str)                                                         \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                 \
        }                                                                              \
    } while (false);

#define DIOPI_CHECK_NULLPTR_ABORT(variable)     \
    do {                                  \
        if (variable == nullptr) {                                                                 \
            printf("The variable `" #variable "` is not defined at %s:%d ", __FILE__, __LINE__);     \
            abort(); \
        }                                                                 \
    } while (false);

#define DIOPI_CALL(Expr)           \
    do {                           \
        diopiError_t ret = Expr;   \
        if (diopiSuccess != ret) { \
            return ret;            \
        }                          \
    } while (false);


enum class MemoryFormat : size_t {
    Contiguous      = 0,
    ChannelsLast    = 1,
    ChannelsLast3d  = 2,
    Preserve        = 3
};

template<typename TensorType>
struct DataType;

template <>
struct DataType<diopiTensorHandle_t> {
    using type = void*;

    static void* data(diopiTensorHandle_t& tensor) {
        void* data;
        diopiGetTensorData(tensor, &data);
        return data;
    }
};

template <>
struct DataType<diopiConstTensorHandle_t> {
    using type = const void*;
    static const void* data(diopiConstTensorHandle_t& tensor) {
        const void* data;
        diopiGetTensorDataConst(tensor, &data);
        return data;
    }
};

template<typename TensorType>
class DiopiTensor final {
public:
    explicit DiopiTensor(TensorType& tensor) : tensor_(tensor) {
        if (tensor_ != nullptr) {
            diopiSize_t diopiShape;
            diopiSize_t diopiStride;
            diopiGetTensorShape(tensor_, &diopiShape);
            std::vector<int32_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
            diopiGetTensorStride(tensor_, &diopiStride);
            std::vector<int32_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
            shape_ = std::move(shapeTmp);
            stride_ = std::move(strideTmp);
        }
    }

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

    const std::vector<int32_t>& shape() {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return shape_;
    }
    const std::vector<int32_t>& stride() {
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
    int64_t dim() {
        return this->shape().size();
    }
    DiopiTensor<diopiTensorHandle_t> contiguous(diopiContextHandle_t ctx, MemoryFormat format = MemoryFormat::Contiguous) {
        /* Returns a new Tensor in new memory format, without data copy */
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
        diopiSize_t diopi_stride(strides.data(), static_cast<int64_t>(strides.size()));
        diopiSize_t diopi_shape;
        diopiGetTensorShape(tensor_, &diopi_shape);
        diopiTensorHandle_t tensor;
        diopiRequireTensor(ctx, &tensor, &diopi_shape, &diopi_stride, this->dtype(), this->device());
        return DiopiTensor<diopiTensorHandle_t>(tensor);
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

    typename DataType<TensorType>::type data() { return DataType<TensorType>::data(tensor_); }

protected:
    TensorType tensor_;
    std::vector<int32_t> shape_;
    std::vector<int32_t> stride_;
};

template <typename TensorType>
inline auto makeTensor(TensorType& tensor) -> DiopiTensor<TensorType> {
    return DiopiTensor<TensorType>(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return makeTensor(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresBuffer(diopiContextHandle_t ctx, int64_t num_bytes) {
    diopiTensorHandle_t tensor;
    diopiRequireBuffer(ctx, &tensor, num_bytes, diopi_device);
    return makeTensor(tensor);
}

inline cnrtQueue_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    return static_cast<cnrtQueue_t>(stream_handle);
}

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_DIOPI_HELPER_HPP_
