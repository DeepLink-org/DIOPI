/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <set>

#include "common.hpp"

namespace impl {
namespace camb {


DiopiTensorT dataTypeCast(diopiContextHandle_t& ctx, const DiopiTensorT& src, diopiDtype_t destDtype) {
    if (src.dtype() == destDtype) {
        return src;
    }
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiSize_t srcSize = vec2diopiSize_t(src.shape());
    DiopiTensorT dest = requiresTensor(ctx, srcSize, destDtype);
    diopiDtype_t srcDtype = src.dtype();
    cnnlCastDataType_t cnnlCastDtype = gCnnlCastDataTypeMapping[{srcDtype, destDtype}];
    DIOPI_CHECK_ABORT(cnnlCastDtype != 0, "data type cast from %d to %d in cnnl is not allown", srcDtype, destDtype);
    CnnlTensorDesc descSrc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc descDest(dest, CNNL_LAYOUT_ARRAY);
    DIOPI_CHECKCNNL(cnnlCastDataType(handle, descSrc.get(), const_cast<DiopiTensorT&>(src).data(), cnnlCastDtype, descDest.get(), dest.data()));
    return dest;
}

void dataTypeCast(diopiContextHandle_t ctx, DiopiTensorT& dest, const DiopiTensorT& src) {
    if (dest.dtype() == src.dtype()) {
        return;
    }
    // check size of dest and src
    assert((void("the shapes of src and dest are not equal"), src.shape() == dest.shape()));
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiDtype_t srcDtype = src.dtype();
    diopiDtype_t destDtype = dest.dtype();
    CnnlTensorDesc descSrc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc descDest(dest, CNNL_LAYOUT_ARRAY);
    cnnlCastDataType_t cnnlCastDtype = gCnnlCastDataTypeMapping[{srcDtype, destDtype}];
    DIOPI_CHECKCNNL(cnnlCastDataType(handle, descSrc.get(), const_cast<DiopiTensorT&>(src).data(), cnnlCastDtype, descDest.get(), dest.data()));
    return;
}

diopiDtype_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes) {
    if (opSupportedDtypes.find(diopi_dtype_float32) != opSupportedDtypes.end()) {
        return diopi_dtype_float32;
    }
    if (opSupportedDtypes.find(diopi_dtype_float16) != opSupportedDtypes.end()) {
        return diopi_dtype_float16;
    }
    if (opSupportedDtypes.find(diopi_dtype_int32) != opSupportedDtypes.end()) {
        return diopi_dtype_int32;
    }
    if (opSupportedDtypes.find(diopi_dtype_int16) != opSupportedDtypes.end()) {
        return diopi_dtype_int16;
    }
    if (opSupportedDtypes.find(diopi_dtype_int8) != opSupportedDtypes.end()) {
        return diopi_dtype_int8;
    }
    if (opSupportedDtypes.find(diopi_dtype_bool) != opSupportedDtypes.end()) {
        return diopi_dtype_bool;
    }
    assert((void("this operator does not support bool, int8, int16, int32, float16, float32"), false));
    return diopi_dtype_int64;  // just for return a value
}

void autoCastTensorType(diopiContextHandle_t ctx, std::vector<DiopiTensorT*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype) {
    // std::multimap<diopiDtype_t, DiopiTensorT*> dtypeAndTensorPtrs;
    std::set<diopiDtype_t> dtypeAndTensorPtrs;
    diopiDtype_t targetType = diopi_dtype_float32;
    for (const auto& pTensor : pTensors) {
        dtypeAndTensorPtrs.insert(pTensor->dtype());
    }
    if (dtypeAndTensorPtrs.find(diopi_dtype_bool) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_bool) == opSupportedDtype.end()) {  // not support bool
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into bool
            targetType = diopi_dtype_bool;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_float64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_float32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float32) == opSupportedDtype.end()) {  // not support float32
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into float32
            targetType = diopi_dtype_float32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_float16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_float16) == opSupportedDtype.end()) {  // not support float16
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into float16
            targetType = diopi_dtype_float16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_int32) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint64) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint32) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int32) == opSupportedDtype.end()) {  // not support int32
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into int32
            targetType = diopi_dtype_int32;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int16) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint16) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int16) == opSupportedDtype.end()) {  // not support int16
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into int16
            targetType = diopi_dtype_int16;
        }
    } else if (dtypeAndTensorPtrs.find(diopi_dtype_int8) != dtypeAndTensorPtrs.end() ||
               dtypeAndTensorPtrs.find(diopi_dtype_uint8) != dtypeAndTensorPtrs.end()) {
        if (opSupportedDtype.find(diopi_dtype_int8) == opSupportedDtype.end()) {  // not support int8
            targetType = choiceDtype(opSupportedDtype);
        } else {  // all tensors cast into int8
            targetType = diopi_dtype_int8;
        }
    } else {
        assert((void("tensor's dtype error, can't be cast"), false));
    }
    for (auto pTensor : pTensors) {
        *pTensor = dataTypeCast(ctx, *pTensor, targetType);
    }
}

}  // namespace camb
}  // namespace impl
