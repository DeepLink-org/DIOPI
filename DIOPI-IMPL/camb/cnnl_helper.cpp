#include "cnnl_helper.hpp"

#include "error.hpp"

diopiError_t convertType(cnnlDataType_t *cnnlType, diopiDtype_t type) {
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
        case diopi_dtype_float64:
            *cnnlType = CNNL_DTYPE_DOUBLE;
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
            impl::camb::set_last_error_string("unkown diopitype error %d at %s:%d", type, __FILE__, __LINE__);
            return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

CnnlHandlePool cnnlHandlePool;
