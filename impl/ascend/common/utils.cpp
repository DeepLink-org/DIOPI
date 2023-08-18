#include "acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t fillTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* out, float val) {
    diopiScalar_t valScalar;
    valScalar.stype = diopi_dtype_float64;
    valScalar.fval = val;
    diopiFill(ctx, *out, &valScalar);
    return diopiSuccess;
}

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype, diopiDevice_t device) {
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    diopiTensorHandle_t outCopy;
    diopiRequireTensor(ctx, &outCopy, &sSize, nullptr, dtype, diopi_host);
    void* ptr;
    diopiGetTensorData(outCopy, &ptr);
    switch (dtype) {
        case diopiDtype_t::diopi_dtype_float32:
            reinterpret_cast<float*>(ptr)[0] = getValue<float>(scalar);
            break;
        case diopiDtype_t::diopi_dtype_float64:
            reinterpret_cast<double*>(ptr)[0] = getValue<double>(scalar);
            break;
        case diopiDtype_t::diopi_dtype_int32:
            reinterpret_cast<int*>(ptr)[0] = getValue<int>(scalar);
            break;
        case diopiDtype_t::diopi_dtype_int64:
            reinterpret_cast<int64_t*>(ptr)[0] = getValue<int64_t>(scalar);
            break;
        case diopiDtype_t::diopi_dtype_uint8:
            reinterpret_cast<uint8_t*>(ptr)[0] = getValue<uint8_t>(scalar);
            break;
        case diopiDtype_t::diopi_dtype_int8:
            reinterpret_cast<int8_t*>(ptr)[0] = getValue<int8_t>(scalar);
            break;
        case diopiDtype_t::diopi_dtype_bool:
            reinterpret_cast<bool*>(ptr)[0] = getValue<bool>(scalar);
            break;
        default:
            error("dtype %d not supported", dtype);
    }
    if (device == diopi_host) {
        *out = outCopy;
    } else {
        int64_t elemsize;
        diopiStreamHandle_t stream;
        diopiGetTensorElemSize(outCopy, &elemsize);
        diopiGetStream(ctx, &stream);
        void *src, *dst;
        diopiRequireTensor(ctx, out, &sSize, nullptr, dtype, diopi_device);
        diopiGetTensorData(*out, &dst);
        diopiGetTensorData(outCopy, &src);
        CALL_ACLRT(aclrtMemcpyAsync(dst, elemsize, src, elemsize, ACL_MEMCPY_HOST_TO_DEVICE, stream));
        CALL_ACLRT(aclrtSynchronizeStream(stream));
    }
    return diopiSuccess;
}

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDevice_t device) {
    return makeTensorFromScalar(ctx, scalar, out, scalar->stype, device);
}

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out, diopiDtype_t dtype) {
    int64_t len = size->getLen();
    int64_t sizeTmp[1] = {len};
    diopiSize_t sSize(sizeTmp, 1);
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
        } else if (dtype == diopi_dtype_int64) {
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
    fillTensor(ctx, out, 1);
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
    }
    check_args(false, "acl not support dioptDtype_t:%d", type);
    return ACL_DT_UNDEFINED;
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

class Element {
public:
    int64_t value;
    int64_t index;
    Element(int value, int index) : value(value), index(index) {}
};

// Custom comparator to sort elements based on their values
bool compare(Element& a, Element& b) { return a.value > b.value; }

std::vector<int64_t> getBaseShape(diopiConstTensorHandle_t src) {
    std::vector<int64_t> baseShapeVec;
    diopiSize_t shape;
    diopiGetTensorShape(src, &shape);
    if (isContiguous(src)) {
        if (shape.getLen() > 0) {
            baseShapeVec.resize(shape.getLen());
            for (int64_t i = 0; i < shape.getLen(); i++) {
                baseShapeVec[i] = shape.data[i];
            }
        } else {
            baseShapeVec.push_back(1);
        }

    } else {
        diopiSize_t stride;
        diopiGetTensorStride(src, &stride);
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < stride.getLen(); i++) {
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
        if (shape.getLen() > 0) {
            diopiGetTensorNumel(src, &numel);
            return numel * elemsize;
        } else {
            return elemsize;
        }
    } else {
        diopiSize_t stride;
        diopiGetTensorStride(src, &stride);
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < stride.getLen(); i++) {
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
    if (isContiguous(src)) {
        diopiCopyInp(ctx, src, srcClone);
    } else {
        const void* data;
        diopiGetTensorDataConst(src, &data);
        info("src: %s", dumpTensor(src).c_str());
        auto baseShapeVec = getBaseShape(src);
        AclOpRunner<4, 1>("AsStrided", ctx)
            .addInput(data, getBaseBufferSize(src), baseShapeVec, ACL_FORMAT_ND, dtype)
            .addConstInput(size)
            .addConstInput(stride)
            .addConstInput(0, diopi_dtype_int64)
            .addOutput(srcClone)
            .run();
    }
    return srcClone;
}

diopiTensorHandle_t contiguous(diopiContextHandle_t cxt, diopiConstTensorHandle_t src) {
    if (isContiguous(src)) {
        return const_cast<diopiTensorHandle_t>(src);
    } else {
        return clone(cxt, src);
    }
}

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiDtype_t dtype) {
    diopiDtype_t srcType;
    diopiGetTensorDtype(src, &srcType);
    if (srcType == dtype) {
        return contiguous(ctx, src);
    }
    diopiTensorHandle_t out;
    if (isContiguous(src)) {
        makeTensorLike(ctx, &out, src, dtype);
        diopiCastDtype(ctx, out, src);
    } else {
        diopiTensorHandle_t outTemp = contiguous(ctx, src);
        makeTensorLike(ctx, &out, outTemp, dtype);
        diopiCastDtype(ctx, out, outTemp);
    }
    return out;
}
}  // namespace ascend
}  // namespace impl
