#include "acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out) {
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    float val = static_cast<float>(scalar->ival);
    diopiRequireTensor(ctx, out, &sSize, nullptr, scalar->stype, diopi_device);
    AclOpRunner<1, 1>("Fills").addInput(*out).setAttr<float>("value", val).addOutput(*out).run(ctx);
    return diopiSuccess;
}

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype) {
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize(sizeTmp, 1);
    float val = static_cast<float>(scalar->ival);
    diopiRequireTensor(ctx, out, &sSize, nullptr, dtype, diopi_device);
    AclOpRunner<1, 1>("Fills").addInput(*out).setAttr<float>("value", val).addOutput(*out).run(ctx);
    return diopiSuccess;
}

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out) {
    int64_t len = size->getLen();
    int64_t buffersize = len * aclDataTypeSize(ACL_INT32);
    int64_t sizeTmp[1] = {len};
    diopiSize_t sSize(sizeTmp, 1);
    diopiRequireTensor(ctx, out, &sSize, nullptr, diopi_dtype_int32, diopi_device);
    if (len > 0) {
        void* ptr;
        CALL_ACLRT(aclrtMallocHost(&ptr, buffersize));
        for (int i = 0; i < len; i++) {
            reinterpret_cast<int32_t*>(ptr)[i] = (int32_t)size->data[i];
        }
        diopiTensorCopyFromBuffer(ctx, const_cast<void*>(ptr), *out);
    }
    return diopiSuccess;
}

aclDataType getAclDataType(diopiConstTensorHandle_t th) {
    diopiDtype_t type;
    diopiGetTensorDtype(th, &type);
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
}  // namespace ascend
}  // namespace impl
