/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cnrt.h>

#include <memory>
#include <set>
#include <unordered_map>

#include "../cnnl_helper.hpp"
#include "common.hpp"

namespace impl {
namespace camb {

#define MAKE_KEY(a, b) (((static_cast<uint64_t>(a) & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(b) & 0xFFFFFFFF))

bool isComplexDtype(diopiDtype_t dtype) { return (dtype == diopi_dtype_complex32 || dtype == diopi_dtype_complex64 || dtype == diopi_dtype_complex128); }

inline bool canCastByInt32(uint64_t castType) {
    // special convert (cnnl doesn't support)
    constexpr uint64_t boolInt8 = MAKE_KEY(diopi_dtype_bool, diopi_dtype_int8);
    constexpr uint64_t boolInt64 = MAKE_KEY(diopi_dtype_bool, diopi_dtype_int64);
    constexpr uint64_t int16Int64 = MAKE_KEY(diopi_dtype_int16, diopi_dtype_int64);
    constexpr uint64_t uint8Bool = MAKE_KEY(diopi_dtype_uint8, diopi_dtype_bool);
    constexpr uint64_t int16Bool = MAKE_KEY(diopi_dtype_int16, diopi_dtype_bool);
    constexpr uint64_t int64Bool = MAKE_KEY(diopi_dtype_int64, diopi_dtype_bool);
    constexpr uint64_t int8Bool = MAKE_KEY(diopi_dtype_int8, diopi_dtype_bool);
    constexpr uint64_t int8Int64 = MAKE_KEY(diopi_dtype_int8, diopi_dtype_int64);
    constexpr uint64_t int64Int8 = MAKE_KEY(diopi_dtype_int64, diopi_dtype_int8);

    return boolInt64 == castType || int16Int64 == castType || uint8Bool == castType || int16Bool == castType || int64Bool == castType || int8Bool == castType ||
           int8Int64 == castType || int64Int8 == castType || boolInt8 == castType;
}

inline bool canCastByFloat32(uint64_t castType) {
    constexpr uint64_t int64Float64 = MAKE_KEY(diopi_dtype_int64, diopi_dtype_float64);
    constexpr uint64_t float64Int64 = MAKE_KEY(diopi_dtype_float64, diopi_dtype_int64);
    constexpr uint64_t uint8Int16 = MAKE_KEY(diopi_dtype_uint8, diopi_dtype_int16);
    constexpr uint64_t int16Uint8 = MAKE_KEY(diopi_dtype_int16, diopi_dtype_uint8);
    constexpr uint64_t uint8Int8 = MAKE_KEY(diopi_dtype_uint8, diopi_dtype_int8);
    constexpr uint64_t int8Uint8 = MAKE_KEY(diopi_dtype_int8, diopi_dtype_uint8);
    constexpr uint64_t int32Uint8 = MAKE_KEY(diopi_dtype_int32, diopi_dtype_uint8);
    constexpr uint64_t uint8Int32 = MAKE_KEY(diopi_dtype_uint8, diopi_dtype_int32);
    constexpr uint64_t float64Uint8 = MAKE_KEY(diopi_dtype_float64, diopi_dtype_uint8);
    constexpr uint64_t uint8Float64 = MAKE_KEY(diopi_dtype_uint8, diopi_dtype_float64);
    constexpr uint64_t float64Int8 = MAKE_KEY(diopi_dtype_float64, diopi_dtype_int8);
    constexpr uint64_t int8Float64 = MAKE_KEY(diopi_dtype_int8, diopi_dtype_float64);

    return int64Float64 == castType || float64Int64 == castType || uint8Int16 == castType || int16Uint8 == castType || uint8Int8 == castType ||
           int8Uint8 == castType || int32Uint8 == castType || uint8Int32 == castType || float64Uint8 == castType || uint8Float64 == castType ||
           float64Int8 == castType || int8Float64 == castType;
}

static diopiError_t dataTypeCastTwice(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src) {
    diopiDtype_t srcDtype = src.dtype();
    diopiDtype_t destDtype = dest.dtype();
    // cast through middle
    auto key = MAKE_KEY(srcDtype, destDtype);
    if (canCastByInt32(key)) {
        DiopiTensor mid = requiresTensor(ctx, src.shape(), diopi_dtype_int32);
        DIOPI_CALL(dataTypeCast(ctx, mid, src));
        DIOPI_CALL(dataTypeCast(ctx, dest, mid));
    } else if (canCastByFloat32(key)) {
        DiopiTensor mid = requiresTensor(ctx, src.shape(), diopi_dtype_float32);
        DIOPI_CALL(dataTypeCast(ctx, mid, src));
        DIOPI_CALL(dataTypeCast(ctx, dest, mid));
    } else {
        // TODO(waiting for dispatch) : cast through cpu
        setLastErrorString(
            "Can not cast from %s to %s at %s:%d ", DiopiDataType::dataTypeStr(srcDtype), DiopiDataType::dataTypeStr(destDtype), __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

#undef MAKE_KEY

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& src, diopiDtype_t destDtype) {
    if (src.dtype() == destDtype) {
        return diopiSuccess;
    }
    DiopiTensor dest = requiresTensor(ctx, src.shape(), destDtype);
    DIOPI_CALL(dataTypeCast(ctx, dest, src));
    src = dest;
    return diopiSuccess;
}

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src) {
    // check size of dest and src
    DIOPI_CHECK(src.shape() == dest.shape(), "the shapes of src and dest are not equal");

    if (dest.dtype() == src.dtype()) {
        if (dest.data() != src.data()) {
            clone(ctx, src, dest);
        }
        return diopiSuccess;
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor destTmp = dest;
    DiopiTensor srcTmp = src;
    // camb only support CNNL_DTYPE_COMPLEX_HALF and CNNL_DTYPE_FLOAT so far
    if (isComplexDtype(src.dtype()) && isComplexDtype(dest.dtype())) {
        destTmp = dest.viewAsReal();
        srcTmp = src.viewAsReal();
    }

    diopiDtype_t srcTmpDtype = srcTmp.dtype();
    diopiDtype_t destTmpDtype = destTmp.dtype();

    auto it = gCnnlCastDataTypeMapping.find({srcTmpDtype, destTmpDtype});
    if (it != gCnnlCastDataTypeMapping.end()) {
        CnnlTensorDesc srcTmpDesc(srcTmp, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc destTmpDesc(destTmp, CNNL_LAYOUT_ARRAY);
        cnnlCastDataType_t castType = it->second;
        DIOPI_CALL_CNNL(cnnlCastDataType(handle, srcTmpDesc.get(), srcTmp.data(), castType, destTmpDesc.get(), destTmp.data()));
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
    } else if (opSupportedDtypes.find(diopi_dtype_complex64) != opSupportedDtypes.end()) {
        *dtype = diopi_dtype_complex64;
    } else {
        setLastErrorString("%s", "this operator does not support bool, int8, int16, int32, float16, float32");
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
        DIOPI_CALL(dataTypeCast(ctx, *pTensor, targetType));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
