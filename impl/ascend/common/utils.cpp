#include "acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype, diopiDevice_t device) {
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    diopiRequireTensor(ctx, out, &sSize, nullptr, dtype, device);
    if (device == diopi_host) {
        void* ptr;
        diopiGetTensorData(*out, &ptr);
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
        }
    } else {
        float val = getValue<float>(scalar);
        AclOpRunner<1, 1>("Fills", ctx).addInput(*out).setAttr<float>("value", val).addOutput(*out).run();
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
}

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(src, &dtype);
    makeTensorLike(ctx, out, src, dtype);
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
    }
    check_args(false, "acl not support dioptDtype_t:%d", type);
    return ACL_DT_UNDEFINED;
}

aclDataType getAclDataType(diopiConstTensorHandle_t th) {
    diopiDtype_t type;
    diopiGetTensorDtype(th, &type);
    return getAclDataType(type);
}
}  // namespace ascend
}  // namespace impl
