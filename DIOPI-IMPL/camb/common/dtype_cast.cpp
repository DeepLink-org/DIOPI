/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cnrt.h>

#include <memory>
#include <set>

#include "common.hpp"

namespace impl {
namespace camb {

#define _MAKE_KEY(a, b) (((static_cast<uint64_t>(a) & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(b) & 0xFFFFFFFF))

inline bool canCastByInt32(uint64_t castType) {
    // special convert (cnnl doesn't support)
    constexpr uint64_t BoolInt64 = _MAKE_KEY(diopi_dtype_bool, diopi_dtype_int64);
    constexpr uint64_t Int16Int64 = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_int64);
    constexpr uint64_t Uint8Bool = _MAKE_KEY(diopi_dtype_uint8, diopi_dtype_bool);
    constexpr uint64_t Int16Bool = _MAKE_KEY(diopi_dtype_int16, diopi_dtype_bool);
    constexpr uint64_t Int64Bool = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_bool);
    constexpr uint64_t Int8Bool = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_bool);
    constexpr uint64_t Int8Int64 = _MAKE_KEY(diopi_dtype_int8, diopi_dtype_int64);
    constexpr uint64_t Int64Int8 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_int8);

    return BoolInt64 == castType || Int16Int64 == castType || Uint8Bool == castType || Int16Bool == castType || Int64Bool == castType || Int8Bool == castType ||
           Int8Int64 == castType || Int64Int8 == castType;
}

inline bool canCastByFloat32(uint64_t castType) {
    constexpr uint64_t Int64Float64 = _MAKE_KEY(diopi_dtype_int64, diopi_dtype_float64);
    constexpr uint64_t Float64Int64 = _MAKE_KEY(diopi_dtype_float64, diopi_dtype_int64);
    return Int64Float64 == castType || Float64Int64 == castType;
}

static diopiError_t dataTypeCastTwice(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiDtype_t srcDtype = src.dtype();
    diopiDtype_t destDtype = dest.dtype();
    cnnlCastDataType_t cast_type;
    // cast through middle
    auto key = _MAKE_KEY(srcDtype, destDtype);
    if (canCastByInt32(key)) {
        DiopiTensor mid = requiresTensor(ctx, src.shape(), diopi_dtype_int32);
        DIOPI_CALL(dataTypeCast(ctx, mid, src));
        DIOPI_CALL(dataTypeCast(ctx, dest, mid));
    } else if (canCastByFloat32) {
        DiopiTensor mid = requiresTensor(ctx, src.shape(), diopi_dtype_float32);
        DIOPI_CALL(dataTypeCast(ctx, mid, src));
        DIOPI_CALL(dataTypeCast(ctx, dest, mid));
    } else {
        // TODO(waiting for dispatch) : cast through cpu
        set_last_error_string("Can not cast from %d to %d at %s:%d ", srcDtype, destDtype, __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

#undef _MAKE_KEY

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& src, diopiDtype_t destDtype) {
    if (src.dtype() == destDtype) {
        return diopiSuccess;
    }
    DiopiTensor dest = requiresTensor(ctx, src.shape(), destDtype);
    dataTypeCast(ctx, dest, src);
    src = dest;
    return diopiSuccess;
}

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src) {
    if (dest.dtype() == src.dtype()) {
        return diopiSuccess;
    }
    // check size of dest and src
    DIOPI_CHECK(src.shape() == dest.shape(), "the shapes of src and dest are not equal");

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiDtype_t srcDtype = src.dtype();
    diopiDtype_t destDtype = dest.dtype();

    auto it = gCnnlCastDataTypeMapping.find({srcDtype, destDtype});
    if (it != gCnnlCastDataTypeMapping.end()) {
        CnnlTensorDesc srcDesc(src, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc destDesc(dest, CNNL_LAYOUT_ARRAY);
        cnnlCastDataType_t castType = it->second;
        DIOPI_CALLCNNL(cnnlCastDataType(handle, srcDesc.get(), src.data(), castType, destDesc.get(), dest.data()));
    } else {
        DIOPI_CALL(dataTypeCastTwice(ctx, dest, src));
    }
    return diopiSuccess;
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
    } else {
        set_last_error_string("%s", "this operator does not support bool, int8, int16, int32, float16, float32");
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype) {
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
    } else {
        set_last_error_string("%s", "tensor's dtype error, can't be cast");
        return diopiDtypeNotSupported;
    }
    for (const auto& pTensor : pTensors) {
        DIOPI_CALL(dataTypeCast(ctx, *pTensor, targetType));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
