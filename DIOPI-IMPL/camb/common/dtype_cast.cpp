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

diopiError_t dataTypeCast(diopiContextHandle_t& ctx, DiopiTensor& src, diopiDtype_t destDtype) {
    if (src.dtype() == destDtype) {
        return diopiSuccess;
    }
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiSize_t srcSize = vec2diopiSize_t(src.shape());
    DiopiTensor dest = requiresTensor(ctx, srcSize, destDtype);
    diopiDtype_t srcDtype = src.dtype();
    if (gCnnlCastDataTypeMapping.find({srcDtype, destDtype}) == gCnnlCastDataTypeMapping.end()) {
        set_last_error_string("can't dtype cast from %s to %s is not allown at %s:%d",
                              DiopiDataType::dataTypeStr(srcDtype).c_str(),
                              DiopiDataType::dataTypeStr(destDtype).c_str(),
                              __FILE__,
                              __LINE__);
        return diopiDtypeNotSupported;
    }
    cnnlCastDataType_t cnnlCastDtype = gCnnlCastDataTypeMapping.at({srcDtype, destDtype});
    CnnlTensorDesc descSrc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc descDest(dest, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlCastDataType(handle, descSrc.get(), const_cast<DiopiTensor&>(src).data(), cnnlCastDtype, descDest.get(), dest.data()));
    src = dest;
    return diopiSuccess;
}

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src) {
    if (dest.dtype() == src.dtype()) {
        return diopiSuccess;
    }
    // check size of dest and src
    assert((void("the shapes of src and dest are not equal"), src.shape() == dest.shape()));
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiDtype_t srcDtype = src.dtype();
    diopiDtype_t destDtype = dest.dtype();
    CnnlTensorDesc descSrc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc descDest(dest, CNNL_LAYOUT_ARRAY);
    if (gCnnlCastDataTypeMapping.find({srcDtype, destDtype}) == gCnnlCastDataTypeMapping.end()) {
        set_last_error_string("can't dtype cast from %s to %s is not allown at %s:%d",
                              DiopiDataType::dataTypeStr(srcDtype).c_str(),
                              DiopiDataType::dataTypeStr(destDtype).c_str(),
                              __FILE__,
                              __LINE__);
        return diopiDtypeNotSupported;
    }
    cnnlCastDataType_t cnnlCastDtype = gCnnlCastDataTypeMapping.at({srcDtype, destDtype});
    DIOPI_CALLCNNL(cnnlCastDataType(handle, descSrc.get(), const_cast<DiopiTensor&>(src).data(), cnnlCastDtype, descDest.get(), dest.data()));
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
        set_last_error_string("this operator does not support bool, int8, int16, int32, float16, float32");
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype) {
    // std::multimap<diopiDtype_t, DiopiTensor*> dtypeAndTensorPtrs;
    std::set<diopiDtype_t> dtypeAndTensorPtrs;
    diopiDtype_t targetType = diopi_dtype_float32;
    for (const auto& pTensor : pTensors) {
        dtypeAndTensorPtrs.insert(pTensor->dtype());
    }
    if (dtypeAndTensorPtrs.find(diopi_dtype_float64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_float32) != dtypeAndTensorPtrs.end()) {
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
        set_last_error_string("tensor's dtype error, can't be cast");
        return diopiDtypeNotSupported;
    }
    for (const auto& pTensor : pTensors) {
        DIOPI_CALL(dataTypeCast(ctx, *pTensor, targetType));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
