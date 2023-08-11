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
#include <diopi/functions.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "error.hpp"

#define DIOPI_CHECK(cond, fmt, args...)                                                      \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            impl::camb::setLastErrorString(#fmt " at %s:%d.\n", ##args, __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                         \
            return diopiErrorOccurred;                                                       \
        }                                                                                    \
    } while (false);

#define DIOPI_CHECK_NULLPTR_ABORT(variable)                                                       \
    do {                                                                                          \
        if (variable == nullptr) {                                                                \
            printf("The variable `" #variable "` is not wangxing at %s:%d ", __FILE__, __LINE__); \
            printf("%s", impl::camb::cambGetLastErrorString(false));                              \
            abort();                                                                              \
        }                                                                                         \
    } while (false);

#define DIOPI_CHECK_ABORT(cond, fmt, args...)                        \
    do {                                                             \
        if (!(cond)) {                                               \
            printf(#fmt " at %s:%d ", ##args, __FILE__, __LINE__);   \
            printf("%s", impl::camb::cambGetLastErrorString(false)); \
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
    static bool isFloatPoint(diopiDtype_t dtype) { return dtype <= 10 && dtype >= 8 || dtype == 12 || dtype == 13; }
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

class DiopiTensor final {
public:
    DiopiTensor() = default;

    // default shallow copy/assignment, it will not change the address of tensor_
    DiopiTensor(const DiopiTensor&) = default;
    DiopiTensor& operator=(const DiopiTensor&) = default;

    explicit DiopiTensor(const diopiTensorHandle_t& tensor) : tensor_(tensor) {
        if (tensor_ != nullptr) {
            diopiSize_t diopiShape;
            diopiSize_t diopiStride;
            diopiDtype_t diopiDtype;
            diopiGetTensorShape(tensor_, &diopiShape);
            std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
            diopiGetTensorStride(tensor_, &diopiStride);
            std::vector<int64_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
            diopiGetTensorDtype(tensor_, &diopiDtype);
            shape_ = std::move(shapeTmp);
            stride_ = std::move(strideTmp);
            dtype_ = diopiDtype;
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
        return dtype_;
    }

    DiopiTensor& setDtype(diopiDtype_t dtype) {
        dtype_ = dtype;
        return *this;
    }

    DiopiTensor viewAsComplex() const {
        int64_t lastDim = size(-1);
        DIOPI_CHECK_ABORT(2 == lastDim, "last dim of tensor must be 2 when view as complex");
        diopiDtype_t complexDtype = DiopiDataType::realDtype2Complex(dtype());
        std::vector<int64_t> complexShape(shape().begin(), shape().end() - 1);
        std::vector<int64_t> complexStride(stride().begin(), stride().end() - 1);
        for (auto& i : complexStride) {
            i /= 2;
        }
        DiopiTensor complexTensor(tensor_);
        complexTensor.asStrided(complexShape, complexStride).setDtype(complexDtype);
        return complexTensor;
    }

    DiopiTensor viewAsReal() const {
        diopiDtype_t realDtype = DiopiDataType::complexDtype2Real(dtype());
        std::vector<int64_t> realShape(shape());
        realShape.push_back(2);
        std::vector<int64_t> realStride(stride());
        for (auto& i : realStride) {
            i *= 2;
        }
        realStride.push_back(1);
        DiopiTensor realTensor(tensor_);
        realTensor.asStrided(realShape, realStride).setDtype(realDtype);
        return realTensor;
    }

    template <typename T>
    std::vector<T> shape() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return std::vector<T>(shape_.begin(), shape_.end());
    }

    const std::vector<int64_t>& shape() const {
        DIOPI_CHECK_NULLPTR_ABORT(tensor_);
        return shape_;
    }

    int64_t size(int i) const {
        if (i < 0) {
            i = shape_.size() + i;
        }
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
    int64_t dim() const { return static_cast<int64_t>(this->shape().size()); }

    DiopiTensor contiguous(diopiContextHandle_t ctx, MemoryFormat format = MemoryFormat::Contiguous) {
        /* DEPRECATED AND WILL BE REMOVED */
        if (this->isContiguous(format)) return *this;
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
        diopiSize_t strideDiopi(strides.data(), static_cast<int64_t>(strides.size()));
        diopiSize_t shapeDiopi(this->shape().data(), static_cast<int64_t>(this->shape().size()));
        diopiTensorHandle_t tensor = nullptr;
        diopiRequireTensor(ctx, &tensor, &shapeDiopi, &strideDiopi, this->dtype(), this->device());
        return DiopiTensor(tensor);
    }

    bool isContiguous(MemoryFormat format = MemoryFormat::Contiguous) const {
        if (!defined()) {
            return true;
        }
        int64_t stride = 1;
        int64_t dim = this->dim();
        auto strides = this->stride();
        auto shape = this->shape();

        if (format == MemoryFormat::Contiguous) {
            for (int64_t i = dim - 1; i >= 0; i--) {
                const auto& shapeD = shape[i];
                if (shapeD != 1) {
                    if (strides[i] != stride) {
                        return false;
                    }
                }
                stride *= shapeD;
            }
        } else if (format == MemoryFormat::ChannelsLast1d) {
            if (strides.size() != 3) {
                return false;
            }
            for (auto& i : {1, 2, 0}) {
                const auto& shapeD = shape[i];
                if (shapeD != 1) {
                    if (strides[i] != stride) {
                        return false;
                    }
                }
                stride *= shapeD;
            }

        } else if (format == MemoryFormat::ChannelsLast) {
            if (strides.size() != 4) return false;
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
        } else if (format == MemoryFormat::ChannelsLast3d) {
            if (strides.size() != 5) return false;
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
        return true;
    }

    DiopiTensor& asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
        this->shape_ = shape;
        this->stride_ = stride;
        return *this;
    }

    DiopiTensor& unsqueeze(int dim) {
        // Note: `channels_last` tensor uses this will become uncontiguous
        // which is same with pytorch
        auto shape = this->shape();
        auto strides = this->stride();
        int64_t newStride = dim >= this->dim() ? 1 : shape[dim] * strides[dim];
        std::vector<int64_t> newShape(shape.begin(), shape.end());
        std::vector<int64_t> newStrides(strides.begin(), strides.end());

        newShape.insert(newShape.begin() + dim, 1);
        newStrides.insert(newStrides.begin() + dim, newStride);
        this->asStrided(newShape, newStrides);
        return *this;
    }

    bool defined() const { return tensor_ != nullptr; }

    DiopiTensor& view(const std::vector<int64_t> shape) {
        // must be contiguous
        std::vector<int64_t> stride(shape.size());
        this->shape_ = shape;
        stride[shape.size() - 1] = 1;
        for (int j = shape_.size() - 2; j >= 0; j--) {
            stride[j] = stride[j + 1] * shape[j + 1];
        }
        this->stride_ = stride;
        return *this;
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

    MemoryFormat suggestMemoryFormat() {
        // TODO(waiting for dispatch): Performance can be improved by dividing is_contiguous into several funcs
        if (this->isContiguous(MemoryFormat::Contiguous)) {
            return MemoryFormat::Contiguous;
        } else if (this->isContiguous(MemoryFormat::ChannelsLast)) {
            return MemoryFormat::ChannelsLast;
        } else {
            return MemoryFormat::ChannelsLast3d;
        }
    }

    diopiTensorHandle_t tensorHandle() { return tensor_; }

    diopiConstTensorHandle_t tensorHandle() const { return tensor_; }

    bool isSame(DiopiTensor t) { return this->tensorHandle() == t.tensorHandle(); }

protected:
    diopiTensorHandle_t tensor_ = nullptr;
    diopiDtype_t dtype_{diopi_dtype_float32};
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
    diopiSize_t sizeTmp(size.data(), size.size());
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
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
    diopiSize_t sizeTmp(size.data(), size.size());
    diopiSize_t strideTmp(stride.data(), stride.size());
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &sizeTmp, &strideTmp, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiSize_t sizeTmp(size.data(), size.size());
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

inline DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype, MemoryFormat memoryFormat) {
    int64_t dim = size.size();
    std::vector<int64_t> strides(dim);
    int64_t stride = 1;
    if (memoryFormat == MemoryFormat::Contiguous) {
        for (size_t i = dim; i > 0; --i) {
            strides[i - 1] = stride;
            if (size[i - 1] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[i - 1];
            }
        }
    } else if (memoryFormat == MemoryFormat::ChannelsLast1d) {
        DIOPI_CHECK_ABORT(size.size() == 3, "%s", "tensor size should be 3");
        for (auto& k : {1, 2, 0}) {
            strides[k] = stride;
            if (size[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[k];
            }
        }

    } else if (memoryFormat == MemoryFormat::ChannelsLast) {
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
    } else if (memoryFormat == MemoryFormat::ChannelsLast3d) {
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
    } else {
        DIOPI_CHECK_ABORT(false, "memory format not support");
    }
    return requiresTensor(ctx, size, strides, dtype);
}

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
    diopiSize_t diopiSize(sizeIn.data(), sizeIn.size());
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
