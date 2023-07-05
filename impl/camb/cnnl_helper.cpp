/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "cnnl_helper.hpp"

#include <functional>

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
            setLastErrorString("unkown diopitype error %d at %s:%d", type, __FILE__, __LINE__);
            return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}
bool CnnlDataType::isFloatPoint(cnnlDataType_t cnnlDT) {
    return cnnlDT == CNNL_DTYPE_HALF || cnnlDT == CNNL_DTYPE_FLOAT || cnnlDT == CNNL_DTYPE_DOUBLE || cnnlDT == CNNL_DTYPE_COMPLEX_HALF ||
           cnnlDT == CNNL_DTYPE_COMPLEX_FLOAT;
}
bool CnnlDataType::isInteger(cnnlDataType_t cnnlDT) {
    return cnnlDT == CNNL_DTYPE_INT8 || cnnlDT == CNNL_DTYPE_INT16 || cnnlDT == CNNL_DTYPE_INT31 || cnnlDT == CNNL_DTYPE_INT32 || cnnlDT == CNNL_DTYPE_INT64 ||
           cnnlDT == CNNL_DTYPE_UINT8 || cnnlDT == CNNL_DTYPE_UINT16 || cnnlDT == CNNL_DTYPE_UINT32 || cnnlDT == CNNL_DTYPE_UINT64;
}
bool CnnlDataType::isBool(cnnlDataType_t cnnlDT) { return cnnlDT == CNNL_DTYPE_BOOL; }

const std::unordered_map<std::vector<diopiDtype_t>, cnnlCastDataType_t, HashCnnlCastDType> gCnnlCastDataTypeMapping{
    {{diopi_dtype_bool, diopi_dtype_int32}, CNNL_CAST_BOOL_TO_INT32},
    {{diopi_dtype_bool, diopi_dtype_float16}, CNNL_CAST_BOOL_TO_HALF},
    {{diopi_dtype_bool, diopi_dtype_float32}, CNNL_CAST_BOOL_TO_FLOAT},

    {{diopi_dtype_int8, diopi_dtype_int16}, CNNL_CAST_INT8_TO_INT16},
    {{diopi_dtype_int8, diopi_dtype_int32}, CNNL_CAST_INT8_TO_INT32},
    {{diopi_dtype_int8, diopi_dtype_float16}, CNNL_CAST_INT8_TO_HALF},
    {{diopi_dtype_int8, diopi_dtype_float32}, CNNL_CAST_INT8_TO_FLOAT},

    {{diopi_dtype_uint8, diopi_dtype_int32}, CNNL_CAST_UINT8_TO_INT32},
    {{diopi_dtype_uint8, diopi_dtype_int64}, CNNL_CAST_UINT8_TO_INT64},
    {{diopi_dtype_uint8, diopi_dtype_float16}, CNNL_CAST_UINT8_TO_HALF},
    {{diopi_dtype_uint8, diopi_dtype_float32}, CNNL_CAST_UINT8_TO_FLOAT},

    {{diopi_dtype_int16, diopi_dtype_int32}, CNNL_CAST_INT16_TO_INT32},
    {{diopi_dtype_int16, diopi_dtype_float16}, CNNL_CAST_INT16_TO_HALF},
    {{diopi_dtype_int16, diopi_dtype_float32}, CNNL_CAST_INT16_TO_FLOAT},
    // no uint16 cast

    {{diopi_dtype_int32, diopi_dtype_bool}, CNNL_CAST_INT32_TO_BOOL},
    {{diopi_dtype_int32, diopi_dtype_int8}, CNNL_CAST_INT32_TO_INT8},
    {{diopi_dtype_int32, diopi_dtype_int16}, CNNL_CAST_INT32_TO_INT16},
    {{diopi_dtype_int32, diopi_dtype_int64}, CNNL_CAST_INT32_TO_INT64},
    {{diopi_dtype_int32, diopi_dtype_float16}, CNNL_CAST_INT32_TO_HALF},
    {{diopi_dtype_int32, diopi_dtype_float32}, CNNL_CAST_INT32_TO_FLOAT},

    {{diopi_dtype_uint32, diopi_dtype_int64}, CNNL_CAST_UINT32_TO_INT64},
    {{diopi_dtype_uint32, diopi_dtype_uint64}, CNNL_CAST_UINT32_TO_UINT64},

    {{diopi_dtype_int64, diopi_dtype_int32}, CNNL_CAST_INT64_TO_INT32},
    {{diopi_dtype_int64, diopi_dtype_uint32}, CNNL_CAST_INT64_TO_UINT32},
    {{diopi_dtype_int64, diopi_dtype_float16}, CNNL_CAST_INT64_TO_HALF},
    {{diopi_dtype_int64, diopi_dtype_float32}, CNNL_CAST_INT64_TO_FLOAT},

    {{diopi_dtype_uint64, diopi_dtype_uint32}, CNNL_CAST_UINT64_TO_UINT32},

    // CNNL_CAST_HALF_TO_FLOAT_INF = 129, /*!< Converts half to float for amp training. */
    {{diopi_dtype_float16, diopi_dtype_bool}, CNNL_CAST_HALF_TO_BOOL},
    {{diopi_dtype_float16, diopi_dtype_int8}, CNNL_CAST_HALF_TO_INT8},
    {{diopi_dtype_float16, diopi_dtype_uint8}, CNNL_CAST_HALF_TO_UINT8},
    {{diopi_dtype_float16, diopi_dtype_int16}, CNNL_CAST_HALF_TO_INT16},
    {{diopi_dtype_float16, diopi_dtype_int32}, CNNL_CAST_HALF_TO_INT32},
    {{diopi_dtype_float16, diopi_dtype_int64}, CNNL_CAST_HALF_TO_INT64},
    {{diopi_dtype_float16, diopi_dtype_float32}, CNNL_CAST_HALF_TO_FLOAT},

    // CNNL_CAST_FLOAT_TO_HALF_IEEE754 = 219, /*!< Converts float to half for ieee754. */
    {{diopi_dtype_float32, diopi_dtype_bool}, CNNL_CAST_FLOAT_TO_BOOL},
    {{diopi_dtype_float32, diopi_dtype_int8}, CNNL_CAST_FLOAT_TO_INT8},
    {{diopi_dtype_float32, diopi_dtype_uint8}, CNNL_CAST_FLOAT_TO_UINT8},
    {{diopi_dtype_float32, diopi_dtype_int16}, CNNL_CAST_FLOAT_TO_INT16},
    {{diopi_dtype_float32, diopi_dtype_int32}, CNNL_CAST_FLOAT_TO_INT32},
    {{diopi_dtype_float32, diopi_dtype_int64}, CNNL_CAST_FLOAT_TO_INT64},
    {{diopi_dtype_float32, diopi_dtype_float16}, CNNL_CAST_FLOAT_TO_HALF},
    {{diopi_dtype_float32, diopi_dtype_float64}, CNNL_CAST_FLOAT_TO_DOUBLE},

    {{diopi_dtype_float64, diopi_dtype_float32}, CNNL_CAST_DOUBLE_TO_FLOAT},
};

CnnlHandlePool cnnlHandlePool;

diopiError_t cnnlTranspose(
    diopiContextHandle_t& ctx, cnnlHandle_t& handle, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layoutIn, cnnlTensorLayout_t layoutOut) {
    /* DEPRECATED AND WILL BE REMOVED */
    DIOPI_CHECK(in.dtype() == out.dtype(), "the data type of input and output tensor should be the same.");

    std::vector<int> order;
    if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {1, 2, 3, 0};
    } else if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {0, 3, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {2, 3, 1, 0};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {0, 2, 3, 1};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {3, 0, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {3, 2, 0, 1};
    } else {
        DIOPI_CHECK(false,
                    "unkown layout error, layout should be "
                    "in [CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_HWCN]");
    }
    CnnlTensorDesc inDesc(in, layoutIn);
    CnnlTensorDesc outDesc(out, layoutOut);
    CnnlTransposeDescriptor transDesc(order.size(), order.data());
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspaceSize));

    void* workspacePtr = workspaceSize == 0 ? requiresBuffer(ctx, workspaceSize).data() : nullptr;
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(), in.data(), outDesc.get(), out.data(), workspacePtr, workspaceSize));
    return diopiSuccess;
}

}  // namespace camb

}  // namespace impl
