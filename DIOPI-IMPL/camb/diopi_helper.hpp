/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_IMPL_CAMB_DIOPI_HELPER_HPP_
#define DIOPI_IMPL_CAMB_DIOPI_HELPER_HPP_

#include <cnnl.h>
#include <cnrt.h>
#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "error.hpp"

namespace impl {
namespace camb {

#define DIOPI_CHECK(cond, fmt, args...)                                          \
    do {                                                                         \
        if (!(cond)) {                                                           \
            set_last_error_string(#fmt " at %s:%d", ##args, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                           \
        }                                                                        \
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

#define DIOPI_CALL(Expr)                                                                                                                                  \
    do {                                                                                                                                                  \
        diopiError_t ret = Expr;                                                                                                                          \
        if (diopiSuccess != ret) {                                                                                                                        \
            set_last_error_string("%s: %s called by `%s` at %s:%d\n", getDiopiErrorStr(ret), camb_get_last_error_string(), __func__, __FILE__, __LINE__); \
            return ret;                                                                                                                                   \
        }                                                                                                                                                 \
    } while (false);

enum class MemoryFormat : size_t { Contiguous = 0, ChannelsLast = 1, ChannelsLast3d = 2, Preserve = 3 };

class DiopiDataType final {
public:
    static bool isInteger(diopiDtype_t dtype) { return dtype < 8; }
    static bool isFloatPoint(diopiDtype_t dtype) { return dtype <= 10 && dtype >= 8 || dtype == 12 || dtype == 13; }
    static std::string dataTypeStr(diopiDtype_t dtype) {
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
                return "diopi_dtype_uint32";
            case diopi_dtype_uint32:
                return "diopi_dtype_int32";
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
            default:
                set_last_error_string("dtype:%d is not support at %s:%d", dtype, __FILE__, __LINE__);
        }
        return nullptr;
    }
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

    int64_t size(int i) {
        return shape()[i];
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
        /* DEPRECATED AND WILL BE REMOVED */
        if (this->is_contiguous(format)) return *this;
        int64_t dim = this->dim();
        std::vector<int64_t> strides(dim);
        int64_t stride = 1;
        if (format == MemoryFormat::Contiguous) {
            for (size_t i = dim; i > 0; --i) {
                strides[i - 1] = stride;
                if (shape_[i - 1] == 0) {
                    continue;
                }
                if (stride != -1) {
                    stride *= shape_[i - 1];
                }
            }
        } else if (format == MemoryFormat::ChannelsLast) {
            DIOPI_CHECK_ABORT(this->shape().size() == 4, "%s", "tensor size should be 4");
            for (auto k : {1, 3, 2, 0}) {
                strides[k] = stride;
                if (shape_[k] == 0) {
                    continue;
                }
                if (stride != -1) {
                    stride *= shape_[k];
                }
            }
        } else if (format == MemoryFormat::ChannelsLast3d) {
            DIOPI_CHECK_ABORT(this->shape().size() == 5, "%s", "tensor size should be 5");
            for (auto k : {1, 4, 3, 2, 0}) {
                strides[k] = stride;
                if (shape_[k] == 0) {
                    continue;
                }
                if (stride != -1) {
                    stride *= shape_[k];
                }
            }
        }
        diopiSize_t stride_diopi(strides.data(), static_cast<int64_t>(strides.size()));
        diopiSize_t shape_diopi(this->shape().data(), this->shape().size());
        diopiTensorHandle_t tensor = nullptr;
        diopiRequireTensor(ctx, &tensor, &shape_diopi, &stride_diopi, this->dtype(), this->device());
        return DiopiTensor(tensor);
    }

    bool is_contiguous(MemoryFormat format = MemoryFormat::Contiguous) {
        int64_t stride = 1;
        int64_t dim = this->dim();
        auto strides = this->stride();
        auto shape = this->shape();

        if (format == MemoryFormat::Contiguous) {
            for (int64_t i = dim - 1; i >= 0; i--) {
                const auto& shape_d = shape[i];
                if (shape_d != 1) {
                    if (strides[i] != stride) {
                        return false;
                    }
                }
                stride *= shape_d;
            }
        } else if (format == MemoryFormat::ChannelsLast) {
            if (strides.size() != 4) return false;
            for (auto& i : {1, 3, 2, 0}) {
                const auto& shape_d = shape[i];
                if (shape_d != 1) {
                    // shape_d != 1 help dealing with shape like [2, 2048, 1, 1]
                    if (strides[i] != stride) {
                        return false;
                    }
                }
                stride *= shape_d;
            }
        } else if (format == MemoryFormat::ChannelsLast3d) {
            if (strides.size() != 5) return false;
            for (auto& i : {1, 4, 3, 2, 0}) {
                const auto& shape_d = shape[i];
                if (shape_d != 1) {
                    if (strides[i] != stride) {
                        return false;
                    }
                }
                stride *= shape[i];
            }
        }
        return true;
    }

    void as_strided(std::vector<int64_t>& shape, std::vector<int64_t>& stride) {
        this->shape_ = shape;
        this->stride_ = stride;
    }

    void unsqueeze(int dim) {
        // Note: `channels_last` tensor uses this will become uncontiguous
        // which is same with pytorch
        auto shape = this->shape();
        auto strides = this->stride();
        int64_t new_stride = dim >= this->dim() ? 1 : shape[dim] * strides[dim];
        std::vector<int64_t> new_shape(shape.begin(), shape.end());
        std::vector<int64_t> new_strides(strides.begin(), strides.end());

        new_shape.insert(new_shape.begin() + dim, 1);
        new_strides.insert(new_strides.begin() + dim, new_stride);
        this->as_strided(new_shape, new_strides);
    }

    bool defined() const {
        if (tensor_ == nullptr) return false;
        return this->numel() != 0;
    }

    void reshape(const std::vector<int64_t> shape) {
        // must be contiguous
        std::vector<int64_t> stride(shape.size());
        this->shape_ = shape;
        stride[shape.size() - 1] = 1;
        for (int j = shape_.size() - 2; j >= 0; j--) {
            stride[j] = stride[j + 1] * shape[j + 1];
        }
        this->stride_ = stride;
    }

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

    MemoryFormat suggest_memory_format() {
        // TODO(waiting for dispatch): Performance can be improved by dividing is_contiguous into several funcs
        if (this->is_contiguous(MemoryFormat::Contiguous)) {
            return MemoryFormat::Contiguous;
        } else if (this->is_contiguous(MemoryFormat::ChannelsLast)) {
            return MemoryFormat::ChannelsLast;
        } else {
            return MemoryFormat::ChannelsLast3d;
        }
    }

    diopiTensorHandle_t tensorHandle() { return tensor_; }

    diopiConstTensorHandle_t tensorHandle() const { return tensor_; }

    bool is_same(DiopiTensor t) { return this->tensorHandle() == t.tensorHandle(); }

protected:
    diopiTensorHandle_t tensor_ = 0;
    std::vector<int64_t> shape_{0};
    std::vector<int64_t> stride_{0};
};

inline auto makeTensor(diopiContextHandle_t ctx, const diopiScalar_t* pScalar) -> DiopiTensor {
    diopiTensorHandle_t tensor = nullptr;
    std::vector<int64_t> shape{1};
    diopiSize_t size(shape.data(), 1);
    diopiRequireTensor(ctx, &tensor, &size, nullptr, pScalar->stype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor ones(diopiContextHandle_t ctx, std::vector<int64_t> size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiSize_t size_(size.data(), size.size());
    diopiRequireTensor(ctx, &tensor, &size_, nullptr, dtype, diopi_device);
    diopiScalar_t scalar = {dtype, 1.0};
    if (DiopiDataType().isInteger(dtype)) scalar = {dtype, 1};
    diopiFill(ctx, tensor, &scalar);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, diopiDtype_t dtype) {
    diopiSize_t size_(size.data(), size.size());
    diopiSize_t stride_(stride.data(), stride.size());
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size_, &stride_, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiSize_t size_(size.data(), size.size());
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size_, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype, MemoryFormat memory_format) {
    int64_t dim = size.size();
    std::vector<int64_t> strides(dim);
    int64_t stride = 1;
    if (memory_format == MemoryFormat::Contiguous) {
        for (size_t i = dim; i > 0; --i) {
            strides[i - 1] = stride;
            if (size[i - 1] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[i - 1];
            }
        }
    } else if (memory_format == MemoryFormat::ChannelsLast) {
        DIOPI_CHECK_ABORT(size.size() == 4, "%s", "tensor size should be 4");
        // constant array is used here to let
        // compiler fully unroll the loop to get better performance
        for (auto& k : {1, 3, 2, 0}) {
            strides[k] = stride;
            if (size[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[k];
            }
        }
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
        DIOPI_CHECK_ABORT(size.size() == 5, "%s", "tensor size should be 5");
        for (auto& k : {1, 4, 3, 2, 0}) {
            strides[k] = stride;
            if (size[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[k];
            }
        }
    }
    return requiresTensor(ctx, size, strides, dtype);
}

inline DiopiTensor requiresBuffer(diopiContextHandle_t ctx, int64_t num_bytes) {
    diopiTensorHandle_t tensor = nullptr;
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
    return std::vector<T>(size.data, size.data + size.len);
}

inline diopiSize_t vec2diopiSize_t(const std::vector<int64_t>& sizeIn) {
    diopiSize_t diopiSize(sizeIn.data(), sizeIn.size());
    return diopiSize;
}

inline void syncStreamInCtx(const diopiContextHandle_t ctx) {
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

#endif  // DIOPI_IMPL_CAMB_DIOPI_HELPER_HPP_
