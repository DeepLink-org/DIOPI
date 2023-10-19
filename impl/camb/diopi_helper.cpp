/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "diopi_helper.hpp"

namespace impl {
namespace camb {

void getFuncName(const char* expr, char* name) {
    for (int i = 0; i < strlen(expr); ++i) {
        if (expr[i] == '(') {
            name[i] = '\0';
            break;
        }
        name[i] = expr[i];
    }
    return;
}

// DiopiDataType

bool DiopiDataType::isInteger(diopiDtype_t dtype) { return dtype < 8; }

bool DiopiDataType::isFloatPoint(diopiDtype_t dtype) { return (dtype <= 10 && dtype >= 8) || dtype == 12 || dtype == 13; }

diopiDtype_t DiopiDataType::complexDtype2Real(diopiDtype_t complexDtype) {
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

diopiDtype_t DiopiDataType::realDtype2Complex(diopiDtype_t realDtype) {
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

const char* DiopiDataType::dataTypeStr(diopiDtype_t dtype) {
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

// DiopiTensor

DiopiTensor::DiopiTensor(const diopiTensorHandle_t& tensor) : tensor_(tensor) {
    if (tensor_ != nullptr) {
        // fix later
        // DIOPI_CHECK_ABORT(this->device() == diopiDevice_t::diopi_device, "%s", "tensor_ is not on camb device.");
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

diopiDevice_t DiopiTensor::device() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor_, &device);
    return device;
}

diopiDtype_t DiopiTensor::dtype() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    return dtype_;
}

DiopiTensor DiopiTensor::viewAsComplex() const {
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

DiopiTensor DiopiTensor::viewAsReal() const {
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

int64_t DiopiTensor::numel() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    int64_t numel;
    diopiGetTensorNumel(tensor_, &numel);
    return numel;
}
int64_t DiopiTensor::elemsize() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    int64_t elemsize;
    diopiGetTensorElemSize(tensor_, &elemsize);
    return elemsize;
}

bool DiopiTensor::isContiguous(diopiMemoryFormat_t format) const {
    if (!defined()) {
        return true;
    }
    int64_t stride = 1;
    int64_t dim = this->dim();
    auto strides = this->stride();
    auto shape = this->shape();

    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = dim - 1; i >= 0; i--) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
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

    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
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
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
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

DiopiTensor& DiopiTensor::asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
    this->shape_ = shape;
    this->stride_ = stride;
    return *this;
}

DiopiTensor& DiopiTensor::unsqueeze(int dim) {
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

DiopiTensor& DiopiTensor::view(const std::vector<int64_t> shape) {
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

void* DiopiTensor::data() {
    void* p = nullptr;
    diopiGetTensorData(tensor_, &p);
    return p;
}
const void* DiopiTensor::data() const {
    const void* p = nullptr;
    diopiGetTensorDataConst(tensor_, &p);
    return p;
}

// other funcs
DiopiTensor makeTensor(diopiContextHandle_t ctx, const diopiScalar_t* pScalar) {
    diopiTensorHandle_t tensor = nullptr;
    std::vector<int64_t> shape{1};
    diopiSize_t size{shape.data(), 1};
    diopiRequireTensor(ctx, &tensor, &size, nullptr, pScalar->stype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor ones(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
    diopiScalar_t scalar = constructDiopiScalarT(dtype, 1);
    diopiFill(ctx, tensor, &scalar);
    return DiopiTensor(tensor);
}

DiopiTensor zeros(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
    diopiScalar_t scalar = constructDiopiScalarT(dtype, 0);
    diopiFill(ctx, tensor, &scalar);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, diopiDtype_t dtype) {
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiSize_t strideTmp{stride.data(), static_cast<int64_t>(stride.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &sizeTmp, &strideTmp, dtype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype, diopiMemoryFormat_t memoryFormat) {
    int64_t dim = size.size();
    std::vector<int64_t> strides(dim);
    int64_t stride = 1;
    if (memoryFormat == diopiMemoryFormat_t::Contiguous) {
        for (size_t i = dim; i > 0; --i) {
            strides[i - 1] = stride;
            if (size[i - 1] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[i - 1];
            }
        }
    } else if (memoryFormat == diopiMemoryFormat_t::ChannelsLast1d) {
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

    } else if (memoryFormat == diopiMemoryFormat_t::ChannelsLast) {
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
    } else if (memoryFormat == diopiMemoryFormat_t::ChannelsLast3d) {
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

DiopiTensor requiresBuffer(diopiContextHandle_t ctx, int64_t numBytes) {
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireBuffer(ctx, &tensor, numBytes, diopi_device);
    return DiopiTensor(tensor);
}

cnrtQueue_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t streamHandle;
    diopiGetStream(ctx, &streamHandle);
    return static_cast<cnrtQueue_t>(streamHandle);
}

diopiSize_t vec2diopiSizeT(const std::vector<int64_t>& sizeIn) {
    diopiSize_t diopiSize{sizeIn.data(), static_cast<int64_t>(sizeIn.size())};
    return diopiSize;
}

void syncStreamInCtx(diopiContextHandle_t ctx) {
    cnrtQueue_t queue = getStream(ctx);
    cnrtQueueSync(queue);
    return;
}

const char* reductionStr(diopiReduction_t reduction) {
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
