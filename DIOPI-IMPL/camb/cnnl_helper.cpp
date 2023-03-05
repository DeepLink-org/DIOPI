#include "cnnl_helper.hpp"

#include "error.hpp"

namespace impl {
namespace camb {

diopiError_t CnnlDataType::convertToCnnlType(cnnlDataType_t* cnnlType, diopiDtype_t type) {
    switch (type) {
        case diopi_dtype_int8:
            *cnnlType = CNNL_DTYPE_INT8;
            break;
        case diopi_dtype_uint8:
            *cnnlType = CNNL_DTYPE_UINT8;
            break;
        case diopi_dtype_int32:
            *cnnlType = CNNL_DTYPE_INT32;
            break;
        case diopi_dtype_uint32:
            *cnnlType = CNNL_DTYPE_UINT32;
            break;
        case diopi_dtype_float16:
            *cnnlType = CNNL_DTYPE_HALF;
            break;
        case diopi_dtype_float32:
            *cnnlType = CNNL_DTYPE_FLOAT;
            break;
        case diopi_dtype_int16:
            *cnnlType = CNNL_DTYPE_INT16;
            break;
        case diopi_dtype_uint16:
            *cnnlType = CNNL_DTYPE_UINT16;
            break;
        case diopi_dtype_bool:
            *cnnlType = CNNL_DTYPE_BOOL;
            break;
        case diopi_dtype_int64:
            *cnnlType = CNNL_DTYPE_INT64;
            break;
        default:
            set_last_error_string("unkown diopitype error %d at %s:%d", type, __FILE__, __LINE__);
            return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}
bool CnnlDataType::isFloat(cnnlDataType_t cnnlDT) {
    return cnnlDT == CNNL_DTYPE_HALF || cnnlDT == CNNL_DTYPE_FLOAT || cnnlDT == CNNL_DTYPE_DOUBLE || cnnlDT == CNNL_DTYPE_COMPLEX_HALF ||
           cnnlDT == CNNL_DTYPE_COMPLEX_FLOAT;
}
bool CnnlDataType::isInteger(cnnlDataType_t cnnlDT) {
    return cnnlDT == CNNL_DTYPE_INT8 || cnnlDT == CNNL_DTYPE_INT16 || cnnlDT == CNNL_DTYPE_INT31 || cnnlDT == CNNL_DTYPE_INT32 || cnnlDT == CNNL_DTYPE_INT64 ||
           cnnlDT == CNNL_DTYPE_UINT8 || cnnlDT == CNNL_DTYPE_UINT16 || cnnlDT == CNNL_DTYPE_UINT32 || cnnlDT == CNNL_DTYPE_UINT64;
}
bool CnnlDataType::isBool(cnnlDataType_t cnnlDT) { return cnnlDT == CNNL_DTYPE_BOOL; }

CnnlHandlePool cnnlHandlePool;

}  // namespace camb

}  // namespace impl
