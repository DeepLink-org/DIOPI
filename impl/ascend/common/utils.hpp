/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_UTILS_HPP_
#define IMPL_ASCEND_COMMON_UTILS_HPP_
#include <vector>

#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

template <typename srcT, typename dstT>
diopiError_t dataCopy(void* dst, const void* src, int64_t size) {
    const srcT* srcArray = reinterpret_cast<const srcT*>(src);
    dstT* dstArray = reinterpret_cast<dstT*>(dst);

    for (int64_t i = 0; i < size; ++i) {
        dstArray[i] = static_cast<dstT>(srcArray[i]);
    }

    return diopiSuccess;
}

const char* diopiDtypeToStr(const diopiDtype_t dtype);

// Those methods can generate new AscendTensor, so context is needed.
diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const diopiSize_t* size, diopiDtype_t dtype, diopiDevice_t device = diopi_device);

diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const std::vector<int64_t>& shape, const std::vector<int64_t>& stride, diopiDtype_t dtype,
                        diopiDevice_t device);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, AscendTensor& dst, const AscendTensor& src, diopiDtype_t dtype = diopi_dtype_unsupported);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, AscendTensor& dst, const diopiScalar_t* scalar, diopiDevice_t device = diopi_device);

diopiError_t reshape(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst, const std::vector<int64_t>& shape);

diopiError_t contiguous(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

diopiError_t castTensor(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst);

diopiError_t castTensor(diopiContextHandle_t ctx, const std::vector<AscendTensor>& src, std::vector<AscendTensor>& dst, diopiDtype_t supportDtype);

/**
 * @brief Convert the data type of an AscendTensor src to the specified supported data type dtype.
 *
 * @param ctx              diopiContextHandle_t context handle for executing operations
 * @param src              Source AscendTensor object for data type conversion
 * @param dtype            Target data type (supported data type)
 *
 * @return diopiError_t    Returns diopiSuccess if the conversion is successful; otherwise, returns other error codes.
 */
diopiError_t castTensor(diopiContextHandle_t ctx, AscendTensor& src, diopiDtype_t supportDtype);

diopiError_t aclAsStrided(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_UTILS_HPP_
