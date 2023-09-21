/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "mlu_ops_helper.hpp"

#include <functional>

#include "error.hpp"

namespace impl {
namespace camb {

diopiError_t MluOpDataType::convertToMluOpType(mluOpDataType_t* mluOpType, diopiDtype_t type) {
    switch (type) {
        case diopi_dtype_int8:
            *mluOpType = MLUOP_DTYPE_INT8;
            break;
        case diopi_dtype_uint8:
            *mluOpType = MLUOP_DTYPE_UINT8;
            break;
        case diopi_dtype_int32:
            *mluOpType = MLUOP_DTYPE_INT32;
            break;
        case diopi_dtype_uint32:
            *mluOpType = MLUOP_DTYPE_UINT32;
            break;
        case diopi_dtype_float16:
            *mluOpType = MLUOP_DTYPE_HALF;
            break;
        case diopi_dtype_float32:
            *mluOpType = MLUOP_DTYPE_FLOAT;
            break;
        case diopi_dtype_float64:
            *mluOpType = MLUOP_DTYPE_DOUBLE;
            break;
        case diopi_dtype_int16:
            *mluOpType = MLUOP_DTYPE_INT16;
            break;
        case diopi_dtype_uint16:
            *mluOpType = MLUOP_DTYPE_UINT16;
            break;
        case diopi_dtype_bool:
            *mluOpType = MLUOP_DTYPE_BOOL;
            break;
        case diopi_dtype_int64:
            *mluOpType = MLUOP_DTYPE_INT64;
            break;
        case diopi_dtype_complex32:
            *mluOpType = MLUOP_DTYPE_COMPLEX_HALF;
            break;
        case diopi_dtype_complex64:
            *mluOpType = MLUOP_DTYPE_COMPLEX_FLOAT;
            break;
        default:
            setLastErrorString("unkown diopitype error %d at %s:%d", type, __FILE__, __LINE__);
            return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

MluOpHandlePool mluOpHandlePool;

}  // namespace camb

}  // namespace impl
