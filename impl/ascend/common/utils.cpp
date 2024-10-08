/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "utils.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"
#include "acloprunner.hpp"

namespace impl {
namespace ascend {

const char* diopiDtypeToStr(diopiDtype_t dtype) {
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
            return "unsupported dtype";
    }
    return "";
}

std::pair<std::array<std::byte, sizeof(int64_t)>, int64_t> getScalarBytes(const diopiScalar_t* scalar, std::optional<diopiDtype_t> castToDtype) {
    std::array<std::byte, sizeof(int64_t)> bytes{};  // just for store the value with the different type.
    int64_t nbytes = 0;
    auto dtype = castToDtype.value_or(scalar->stype);
#define DIOPI_GET_SCALAR_BYTES_CASE(diopiType, ctype) \
    case diopiType: {                                 \
        nbytes = sizeof(ctype);                       \
        ctype val = getValue<ctype>(scalar);          \
        std::memcpy(bytes.data(), &val, nbytes);      \
        break;                                        \
    }
    switch (dtype) {
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_bool, bool)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_int8, int8_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_uint8, uint8_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_int16, int16_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_uint16, uint16_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_int32, int32_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_uint32, uint32_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_int64, int64_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_uint64, uint64_t)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_float16, half_float::half)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_float32, float)
        DIOPI_GET_SCALAR_BYTES_CASE(diopi_dtype_float64, double)
        default: {
            error(__FILE__, __LINE__, __FUNCTION__, "invalid input tensor dtype: %s", diopiDtypeToStr(scalar->stype));
        }
    }
    return {bytes, nbytes};
}

// ascend tensor utils
diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const diopiSize_t* size, const diopiSize_t* stride, diopiDtype_t dtype,
                        diopiDevice_t device) {
    diopiTensorHandle_t dstPtr;
    diopiRequireTensor(ctx, &dstPtr, size, stride, dtype, device);
    dst = AscendTensor(dstPtr);
    ASCEND_CHECK_ABORT(dst.defined(), "generate Ascend Tensor failed, it's nullptr.");
    return diopiSuccess;
}

diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const diopiSize_t* size, diopiDtype_t dtype, diopiDevice_t device) {
    return makeTensor(ctx, dst, size, nullptr, dtype, device);
}

diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const std::vector<int64_t>& shape, const std::vector<int64_t>& stride, diopiDtype_t dtype,
                        diopiDevice_t device) {
    diopiSize_t shapeTmp{shape.data(), static_cast<int64_t>(shape.size())};
    if (stride.empty()) {
        return makeTensor(ctx, dst, &shapeTmp, nullptr, dtype, device);
    } else {
        diopiSize_t strideTmp{stride.data(), static_cast<int64_t>(stride.size())};
        return makeTensor(ctx, dst, &shapeTmp, &strideTmp, dtype, device);
    }
    return diopiSuccess;
}

diopiError_t makeTensorLike(diopiContextHandle_t ctx, AscendTensor& dst, const AscendTensor& src, diopiDtype_t dtype) {
    if (diopi_dtype_unsupported == dtype) {
        return makeTensor(ctx, dst, src.shape(), src.stride(), src.dtype(), src.device());
    } else {
        return makeTensor(ctx, dst, src.shape(), src.stride(), dtype, src.device());
    }
}

diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const std::vector<int64_t>& shape, diopiDtype_t dtype) {
    return makeTensor(ctx, dst, shape, std::vector<int64_t>{}, dtype, diopi_device);
}

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, AscendTensor& dst, const diopiScalar_t* scalar, diopiDevice_t device) {
    std::vector<int64_t> shape{1};
    makeTensor(ctx, dst, shape, scalar->stype);
    auto th = const_cast<diopiTensorHandle_t>(static_cast<diopiConstTensorHandle_t>(dst));
    if (diopi_device == device) {
        return diopiFill(ctx, th, scalar);
    } else {
        void* ptr;
        diopiGetTensorData(th, &ptr);
        if (isFloatingType(scalar->stype)) {
            reinterpret_cast<double*>(ptr)[0] = getValue<double>(scalar);
        } else {
            reinterpret_cast<int64_t*>(ptr)[0] = getValue<int64_t>(scalar);
        }
        return diopiSuccess;
    }
}

diopiError_t fillNan(diopiContextHandle_t ctx, AscendTensor& src) {
    // get nan value tensor
    diopiTensorHandle_t nanValue;
    auto zeroValueScalar = constructDiopiScalarT(diopi_dtype_float64, 0.0);
    makeTensorFromScalar(ctx, &zeroValueScalar, &nanValue, diopi_dtype_float32, diopi_device);
    diopiDivInpScalar(ctx, nanValue, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);

    diopiTensorHandle_t onePtr;
    makeOnesLike(ctx, &onePtr, src.tensorHandle());
    AscendTensor nan(nanValue), one(onePtr);
    castTensor(ctx, one, diopi_dtype_bool);
    diopiMaskedFillInp(ctx, const_cast<diopiTensorHandle_t>(src.tensorHandle()), one.tensorHandle(), nan.tensorHandle());
    return diopiSuccess;
}

diopiError_t reshape(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst, const std::vector<int64_t>& shape) {
    ASCEND_CHECK_ABORT(src.isContiguous(), "now only contiguous tensor support reshape by shape.");
    if (src.isSame(dst)) {
        dst.view(shape);
        return diopiSuccess;
    }

    // make dst tensor with `shape`
    AscendTensor tmp = src;
    tmp.view(shape);
    makeTensorLike(ctx, dst, tmp);

    auto sourcePtr = const_cast<void*>(src.data());
    auto destPtr = const_cast<void*>(dst.data());
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    aclrtMemcpyAsync(destPtr, dst.getAclMemBufferSize(), sourcePtr, src.getAclMemBufferSize(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);

    return diopiSuccess;
}

AscendTensor reshape(diopiContextHandle_t ctx, const AscendTensor& src, const std::vector<int64_t>& shape) {
    ASCEND_CHECK_ABORT(src.defined(), "input tensor is nullptr.");

    // if shape is the same as src, return src directly.
    if (src.shape() == shape) {
        return src;
    }

    // if shape is not the same as src, create a new tensor, then copy the data from src to the new tensor.
    AscendTensor result, srcCopy(src);
    makeTensor(ctx, result, shape, srcCopy.dtype());
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, result, srcCopy.view(shape));

    return AscendTensor(result.tensorHandle());
}

diopiError_t aclAsStridedCore(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst) {
    diopiTensorHandle_t targetObj = const_cast<diopiTensorHandle_t>(static_cast<diopiConstTensorHandle_t>(dst));
    AclOpRunner<4, 1>("AsStrided", ctx)
        .addInput(src.data(), src.getAclMemBufferSize(), src.getAclMemShape(), src.getAclDataFormat(), src.dtype())
        .addConstInput(src.shape())
        .addConstInput(src.stride())
        .addConstInput(0, diopi_dtype_int64)
        .addOutput(targetObj)
        .run();

    // update Ascend Tensor attribute.
    dst = AscendTensor(targetObj);
    return diopiSuccess;
}

diopiError_t contiguous(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst, diopiMemoryFormat_t format) {
    if (src.isContiguous(format)) {
        dst = const_cast<AscendTensor&>(src);
        return diopiSuccess;
    }

    return aclAsStrided(ctx, src, dst);
}

diopiError_t castTensor(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst) {
    ASCEND_CHECK_ABORT(dst.shape() == src.shape(), "required input and output has the same shape.");
    if (src.data() == dst.data()) {
        return diopiSuccess;
    }

    auto dstPtr = const_cast<diopiTensorHandle_t>(static_cast<diopiConstTensorHandle_t>(dst));
    diopiCastDtype(ctx, dstPtr, static_cast<diopiConstTensorHandle_t>(src));
    dst = AscendTensor(dstPtr);

    return diopiSuccess;
}

diopiError_t castTensor(diopiContextHandle_t ctx, const std::vector<AscendTensor>& src, std::vector<AscendTensor>& dst, diopiDtype_t supportDtype) {
    ASCEND_CHECK_ABORT(src.size() == dst.size(), "require input size equal output size.");
    for (int i = 0; i < src.size(); ++i) {
        CHECK_ASCENDRT(castTensor(ctx, src[i], dst[i]));
    }
    return diopiSuccess;
}

diopiError_t castTensor(diopiContextHandle_t ctx, AscendTensor& src, diopiDtype_t dtype) {
    if (dtype == src.dtype()) {
        return diopiSuccess;
    }
    AscendTensor temp;
    makeTensorLike(ctx, temp, src, dtype);
    castTensor(ctx, src, temp);
    src = temp;
    return diopiSuccess;
}

diopiError_t aclAsStrided(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst) {
    if (src.dtype() != diopi_dtype_float64) {
        return aclAsStridedCore(ctx, src, dst);
    } else {
        AscendTensor srcCpy = const_cast<AscendTensor&>(src);
        castTensor(ctx, srcCpy, diopi_dtype_float32);
        castTensor(ctx, dst, diopi_dtype_float32);

        return aclAsStridedCore(ctx, srcCpy, dst);
    }
}

// diopi tensor utils
diopiError_t fillTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, float val) {
    auto valScalar = constructDiopiScalarT(diopi_dtype_float64, val);
    diopiFill(ctx, out, &valScalar);
    return diopiSuccess;
}

diopiError_t fillTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, int val) {
    auto valScalar = constructDiopiScalarT(diopi_dtype_int64, val);
    diopiFill(ctx, out, &valScalar);
    return diopiSuccess;
}

diopiError_t fillTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double val) {
    auto valScalar = constructDiopiScalarT(diopi_dtype_float64, val);
    diopiFill(ctx, out, &valScalar);
    return diopiSuccess;
}

diopiTensorHandle_t createTensorIfNullptrOrConstCast(diopiContextHandle_t ctx, diopiConstTensorHandle_t in, diopiSize_t& shape, diopiDtype_t dtype,
                                                     bool isFillingRequired, double value) {
    diopiTensorHandle_t out;
    if (nullptr == in) {
        diopiRequireTensor(ctx, &out, &shape, nullptr, dtype, diopi_device);
        if (isFillingRequired) {
            fillTensor(ctx, out, value);
        }
    } else {
        out = const_cast<diopiTensorHandle_t>(in);
    }
    return out;
}

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype, diopiDevice_t device) {
    // get scalar
    auto [bytes, nbytes] = getScalarBytes(scalar, dtype);
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize = arrayToDiopiSize(sizeTmp, 1);
    void* outDataPtr = nullptr;
    diopiTensorHandle_t outTmp = nullptr;
    diopiRequireTensor(ctx, &outTmp, &sSize, nullptr, dtype, device);
    if (device == diopi_host) {
        diopiGetTensorData(outTmp, &outDataPtr);
        memcpy(outDataPtr, bytes.data(), nbytes);
    } else if (device == diopi_device) {
        diopiFill(ctx, outTmp, scalar);
    } else {
        error(__FILE__, __LINE__, __FUNCTION__, "device(%s) not supported", deviceType2Str(device));
    }
    *out = outTmp;
    return diopiSuccess;
}

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDevice_t device) {
    return makeTensorFromScalar(ctx, scalar, out, scalar->stype, device);
}

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out, diopiDtype_t dtype) {
    int64_t len = size->len;
    int64_t sizeTmp[1] = {len};
    diopiSize_t sSize = arrayToDiopiSize(sizeTmp, 1);
    diopiRequireTensor(ctx, out, &sSize, nullptr, dtype, diopi_host);
    if (len > 0) {
        void* dst = nullptr;
        diopiGetTensorData(*out, &dst);
        if (dtype == diopi_dtype_int64) {
            for (int i = 0; i < len; i++) {
                reinterpret_cast<int64_t*>(dst)[i] = (int64_t)size->data[i];
            }
        } else if (dtype == diopi_dtype_int32) {
            for (int i = 0; i < len; i++) {
                reinterpret_cast<int32_t*>(dst)[i] = (int32_t)size->data[i];
            }
        } else if (dtype == diopi_dtype_int16) {
            for (int i = 0; i < len; i++) {
                reinterpret_cast<int16_t*>(dst)[i] = (int16_t)size->data[i];
            }
        } else if (dtype == diopi_dtype_bool) {
            for (int i = 0; i < len; i++) {
                reinterpret_cast<bool*>(dst)[i] = static_cast<bool>(size->data[i]);
            }
        }
    }
    return diopiSuccess;
}

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out) {
    return makeTensorFromSize(ctx, size, out, diopi_dtype_int64);
}

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype) {
    diopiDevice_t device;
    diopiSize_t size, stride;
    diopiGetTensorDevice(src, &device);
    diopiGetTensorShape(src, &size);
    diopiGetTensorStride(src, &stride);
    diopiRequireTensor(ctx, out, &size, &stride, dtype, device);
    return diopiSuccess;
}

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(src, &dtype);
    return makeTensorLike(ctx, out, src, dtype);
}

diopiError_t makeOnesLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype) {
    makeTensorLike(ctx, out, src, dtype);
    fillTensor(ctx, *out, static_cast<float>(1.0));
    return diopiSuccess;
}

diopiError_t makeOnesLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(src, &dtype);
    return makeOnesLike(ctx, out, src, dtype);
}

aclDataType getAclDataType(diopiDtype_t type) {
    switch (type) {
        case diopi_dtype_float16:
            return ACL_FLOAT16;
        case diopi_dtype_float32:
            return ACL_FLOAT;
        case diopi_dtype_float64:
            return ACL_DOUBLE;
        case diopi_dtype_int8:
            return ACL_INT8;
        case diopi_dtype_uint8:
            return ACL_UINT8;
        case diopi_dtype_int16:
            return ACL_INT16;
        case diopi_dtype_uint16:
            return ACL_UINT16;
        case diopi_dtype_int32:
            return ACL_INT32;
        case diopi_dtype_uint32:
            return ACL_UINT32;
        case diopi_dtype_int64:
            return ACL_INT64;
        case diopi_dtype_uint64:
            return ACL_UINT64;
        case diopi_dtype_bool:
            return ACL_BOOL;
        case diopi_dtype_complex64:
            return ACL_COMPLEX64;
        case diopi_dtype_complex128:
            return ACL_COMPLEX128;
        case diopi_dtype_bfloat16:
            return ACL_BF16;
        default:
            ASCEND_CHECK_ABORT(false, "acl not support dioptDtype_t:%d", type);
            return ACL_DT_UNDEFINED;
    }
}

aclDataType getAclDataType(diopiConstTensorHandle_t th) {
    diopiDtype_t type;
    diopiGetTensorDtype(th, &type);
    return getAclDataType(type);
}

bool isContiguous(diopiConstTensorHandle_t tensor, diopiMemoryFormat_t format) {
    diopiSize_t size, strideDiopi;
    diopiGetTensorShape(tensor, &size);
    diopiGetTensorStride(tensor, &strideDiopi);
    auto dim = size.len;
    auto shape = size.data;
    auto strides = strideDiopi.data;
    int64_t stride = 1;

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
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        if (dim != 4) return false;
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
        if (dim != 5) return false;
        for (auto& i : {1, 4, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        if (dim != 3) return false;
        for (auto& i : {1, 2, 0}) {
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

std::vector<int64_t> getBaseShape(diopiConstTensorHandle_t src) {
    std::vector<int64_t> baseShapeVec;
    diopiSize_t shape;
    diopiGetTensorShape(src, &shape);
    if (isContiguous(src)) {
        if (shape.len > 0) {
            baseShapeVec.resize(shape.len);
            for (int64_t i = 0; i < shape.len; i++) {
                baseShapeVec[i] = shape.data[i];
            }
        } else {
            baseShapeVec.push_back(1);
        }

    } else {
        diopiSize_t stride;
        diopiGetTensorStride(src, &stride);
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < stride.len; i++) {
            if (stride.data[i] > maxStride) {
                maxStride = stride.data[i];
                maxIdx = i;
            }
        }
        if (maxStride > 0) {
            baseShapeVec.push_back(shape.data[maxIdx] * maxStride);
        } else {
            baseShapeVec.push_back(1);
        }
    }
    return baseShapeVec;
}

int64_t getBaseBufferSize(diopiConstTensorHandle_t src) {
    int64_t numel = 1, elemsize;
    diopiSize_t shape;
    diopiGetTensorShape(src, &shape);
    diopiGetTensorElemSize(src, &elemsize);
    if (isContiguous(src)) {
        if (shape.len > 0) {
            diopiGetTensorNumel(src, &numel);
            return numel * elemsize;
        } else {
            return elemsize;
        }
    } else {
        diopiSize_t stride;
        diopiGetTensorStride(src, &stride);
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < stride.len; i++) {
            if (stride.data[i] > maxStride) {
                maxStride = stride.data[i];
                maxIdx = i;
            }
        }
        if (maxStride > 0) {
            return shape.data[maxIdx] * maxStride * elemsize;
        } else {
            return elemsize;
        }
    }
}

diopiTensorHandle_t clone(diopiContextHandle_t ctx, diopiConstTensorHandle_t src) {
    diopiTensorHandle_t srcClone;
    diopiSize_t size, stride;
    diopiDtype_t dtype;
    diopiGetTensorDtype(src, &dtype);
    diopiGetTensorShape(src, &size);
    diopiRequireTensor(ctx, &srcClone, &size, nullptr, dtype, diopi_device);
    diopiGetTensorStride(src, &stride);
    // if (isContiguous(src)) {
    ::impl::ascend_npu::diopiCopyInp(ctx, src, srcClone);
    // } else {
    //     AscendTensor srcAt(src), srcCloneAt(srcClone);
    //     aclAsStrided(ctx, srcAt, srcCloneAt);
    //     srcClone = const_cast<diopiTensorHandle_t>(static_cast<diopiConstTensorHandle_t>(srcCloneAt));
    // }
    return srcClone;
}

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiMemoryFormat_t format) {
    if (isContiguous(src, format)) {
        return const_cast<diopiTensorHandle_t>(src);
    } else {
        return clone(ctx, src);
    }
}

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiDtype_t dtype, diopiMemoryFormat_t format) {
    diopiDtype_t srcType;
    diopiGetTensorDtype(src, &srcType);
    if (srcType == dtype) {
        return contiguous(ctx, src, format);
    }
    diopiTensorHandle_t out;
    if (isContiguous(src, format)) {
        makeTensorLike(ctx, &out, src, dtype);
        diopiCastDtype(ctx, out, src);
    } else {
        diopiTensorHandle_t outTemp = contiguous(ctx, src, format);
        makeTensorLike(ctx, &out, outTemp, dtype);
        diopiCastDtype(ctx, out, outTemp);
    }
    return out;
}

diopiSize_t vectorToDiopiSize(std::vector<int64_t>& sizeVec) {
    diopiSize_t size;
    size.len = sizeVec.size();
    size.data = sizeVec.data();
    return size;
}

diopiSize_t arrayToDiopiSize(int64_t* data, int64_t len) {
    diopiSize_t size;
    size.len = len;
    size.data = data;
    return size;
}

diopiError_t transTensorTo2D(diopiContextHandle_t ctx, AscendTensor& th) {
    if (th.shape().size() < 2) return diopiErrorOccurred;
    std::vector<int64_t> dims;
    std::vector<int64_t> thShape = th.shape();
    int dim1 = std::accumulate(thShape.begin(), thShape.end() - 1, 1, std::multiplies<>());
    dims = {dim1, thShape.back()};
    th.view(dims);
    return diopiSuccess;
}

diopiError_t broadcast(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const std::vector<int64_t>& size) {
    AscendTensor atout(out);
    const AscendTensor atinp(input);
    return broadcast(ctx, atout, atinp, size);
}

diopiError_t broadcast(diopiContextHandle_t ctx, AscendTensor& out, const AscendTensor& input, const std::vector<int64_t>& size) {
    if (size.empty()) {
        diopiCastDtype(ctx, const_cast<diopiTensorHandle_t>(out.tensorHandle()), const_cast<diopiTensorHandle_t>(input.tensorHandle()));
        return diopiSuccess;
    }
    // Avoid modifying the input tensor (when input == out).
    AscendTensor tmp = out;
    if (!out.defined() || input.isSame(out)) {
        AscendTensor tmp1;
        makeTensor(ctx, tmp1, size, input.dtype());
        tmp = tmp1;
    }
    auto ptr = const_cast<diopiTensorHandle_t>(tmp.tensorHandle());
    AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(input).addConstInput(size).addOutput(ptr).run();
    out = AscendTensor(ptr);
    return diopiSuccess;
}

std::vector<int64_t> inferSize(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    size_t dimsA = shape1.size();
    size_t dimsB = shape2.size();
    size_t ndim = dimsA > dimsB ? dimsA : dimsB;
    std::vector<int64_t> expandedSizes(ndim);

    // Use ptrdiff_t to ensure signed comparison.
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
        ptrdiff_t offset = ndim - 1 - i;
        ptrdiff_t dimA = dimsA - 1 - offset;
        ptrdiff_t dimB = dimsB - 1 - offset;
        auto sizeA = (dimA >= 0) ? shape1[dimA] : 1;
        auto sizeB = (dimB >= 0) ? shape2[dimB] : 1;

        // 1s map to the other size (even 0).
        expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
    }

    return expandedSizes;
}

diopiTensorHandle_t hostToDevice(diopiContextHandle_t ctx, diopiConstTensorHandle_t src) {
    diopiDevice_t device;
    diopiGetTensorDevice(src, &device);
    if (device == diopi_host) {
        diopiTensorHandle_t dst;
        diopiSize_t size, stride;
        diopiDtype_t dtype;
        diopiGetTensorShape(src, &size);
        diopiGetTensorStride(src, &stride);
        diopiGetTensorDtype(src, &dtype);
        diopiRequireTensor(ctx, &dst, &size, &stride, dtype, diopi_device);
        const void* srcPtr;
        void* dstPtr;
        diopiGetTensorDataConst(src, &srcPtr);
        diopiGetTensorData(dst, &dstPtr);
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        int64_t elemsize = getBaseBufferSize(src);
        CALL_ACLRT(aclrtMemcpyAsync(dstPtr, elemsize, const_cast<void*>(srcPtr), elemsize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        return dst;
    } else {
        return const_cast<diopiTensorHandle_t>(src);
    }
}

AscendTensor hostToDeviceAsync(diopiContextHandle_t ctx, const AscendTensor& hostTensor) {
    diopiDevice_t device = hostTensor.device();

    if (device == diopi_host) {
        diopiTensorHandle_t dst;
        diopiSize_t size{hostTensor.shape().data(), hostTensor.dim()};
        diopiSize_t stride{hostTensor.stride().data(), (int64_t)hostTensor.stride().size()};
        diopiDtype_t dtype = hostTensor.dtype();
        diopiRequireTensor(ctx, &dst, &size, &stride, dtype, diopi_device);
        const void* srcPtr = hostTensor.data();
        void* dstPtr;
        diopiGetTensorData(dst, &dstPtr);
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        int64_t elemsize = hostTensor.numel() * hostTensor.elemsize();
        CALL_ACLRT(aclrtMemcpyAsync(dstPtr, elemsize, const_cast<void*>(srcPtr), elemsize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        return AscendTensor(dst);
    } else {
        return hostTensor;
    }
}

AscendTensor deviceToHostSync(diopiContextHandle_t ctx, const AscendTensor& deviceTensor) {
    if (deviceTensor.device() == diopi_device) {
        diopiTensorHandle_t dst;
        diopiSize_t size{deviceTensor.shape().data(), deviceTensor.dim()};
        diopiSize_t stride{deviceTensor.stride().data(), (int64_t)deviceTensor.stride().size()};
        diopiDtype_t dtype = deviceTensor.dtype();
        diopiRequireTensor(ctx, &dst, &size, &stride, dtype, diopi_host);
        const void* srcPtr = deviceTensor.data();
        void* dstPtr;
        diopiGetTensorData(dst, &dstPtr);
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        int64_t elemsize = deviceTensor.numel() * deviceTensor.elemsize();
        CALL_ACLRT(aclrtMemcpyAsync(dstPtr, elemsize, const_cast<void*>(srcPtr), elemsize, ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CALL_ACLRT(aclrtSynchronizeStream(stream));
        return AscendTensor(dst);
    } else {
        return deviceTensor;
    }
}

static diopiError_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes, diopiDtype_t* dtype) {
    if (opSupportedDtypes.find(diopi_dtype_float32) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_float32;
    } else if (opSupportedDtypes.find(diopi_dtype_float16) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_float16;
    } else if (opSupportedDtypes.find(diopi_dtype_int32) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_int32;
    } else if (opSupportedDtypes.find(diopi_dtype_int16) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_int16;
    } else if (opSupportedDtypes.find(diopi_dtype_int8) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_int8;
    } else if (opSupportedDtypes.find(diopi_dtype_bool) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_bool;
    } else if (opSupportedDtypes.find(diopi_dtype_complex64) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_complex64;
    } else {
        setLastErrorString("%s", "this operator does not support bool, int8, int16, int32, float16, float32");
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<AscendTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype) {
    std::set<diopiDtype_t> dtypeAndTensorPtrs;
    diopiDtype_t targetType = diopi_dtype_float32;
    for (const auto& pTensor : pTensors) {
        dtypeAndTensorPtrs.insert(pTensor->dtype());
    }
    if (dtypeAndTensorPtrs.find(diopi_dtype_float64) != dtypeAndTensorPtrs.end() || dtypeAndTensorPtrs.find(diopi_dtype_float32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float32) == opSupportedDtype.end()) {  // not support float32
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into float32
            targetType = diopi_dtype_float32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_float16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float16) == opSupportedDtype.end()) {  // not support float16
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into float16
            targetType = diopi_dtype_float16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_int32) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int32) == opSupportedDtype.end()) {  // not support int32
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into int32
            targetType = diopi_dtype_int32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int16) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int16) == opSupportedDtype.end()) {  // not support int16
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into int16
            targetType = diopi_dtype_int16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int8) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint8) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int8) == opSupportedDtype.end()) {  // not support int8
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into int8
            targetType = diopi_dtype_int8;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_bool) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_bool) == opSupportedDtype.end()) {  // not support bool
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into bool
            targetType = diopi_dtype_bool;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_complex64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_complex128) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_complex64) == opSupportedDtype.end()) {  // not support bool
            DIOPI_CALL(choiceDtype(opSupportedDtype, &targetType));
        } else {  // all tensors cast into bool
            targetType = diopi_dtype_bool;
        }
    } else {
        setLastErrorString("%s", "tensor's dtype error, can't be cast");
        return diopiDtypeNotSupported;
    }
    for (const auto& pTensor : pTensors) {
        castTensor(ctx, *pTensor, targetType);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
