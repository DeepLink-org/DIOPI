/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "debug.hpp"

#include "utils.hpp"

namespace impl {
namespace ascend {

void printContiguousTensor(diopiContextHandle_t ctx, const AscendTensor& at, char* name) {
    printf("==========[Tensor name %s]==========\n", name);
    if (!at.isContiguous()) {
        printf("input tensor is not contiguous. break;");
        return;
    }
    void* ptrHost;

    printf("Tensor device: %s\n", (at.device() ? "diopi_device" : "diopi_host"));

    if (at.device() == diopiDevice_t::diopi_device) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        CALL_ACLRT(aclrtMallocHost(&ptrHost, at.numel() * at.elemsize()));
        CALL_ACLRT(aclrtMemcpyAsync(
            ptrHost, at.numel() * at.elemsize(), at.data(), at.numel() * at.elemsize(), ACL_MEMCPY_DEVICE_TO_HOST, reinterpret_cast<aclrtStream>(stream)));
        CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));
    } else {
        const void* ptrHostCopy;
        diopiGetTensorDataConst(at.tensorHandle(), &ptrHostCopy);
        ptrHost = const_cast<void*>(ptrHostCopy);
    }

    std::cout << "numel = " << at.numel() << std::endl;

    for (int i = 0; i < at.dim(); ++i) {
        std::cout << "stride(" << i << ") = " << at.stride(i) << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < at.dim(); ++i) {
        std::cout << "shape(" << i << ") = " << at.shape(i) << std::endl;
    }
    printf("Tensor type %d \n", at.dtype());
    printf("Tensor %s:\n\n", name);
    for (int64_t i = 0; i < at.numel(); i++) {
        switch (at.dtype()) {
            case diopi_dtype_float32:
                printf("item %ld: %f\n", i, reinterpret_cast<float*>(ptrHost)[i]);
                break;
            case diopi_dtype_float64:
                printf("item %ld: %f\n", i, reinterpret_cast<double*>(ptrHost)[i]);
                break;
            case diopi_dtype_int32:
                printf("item %ld: %d\n", i, reinterpret_cast<int*>(ptrHost)[i]);
                break;
            case diopi_dtype_int64:
                printf("item %ld: %ld\n", i, reinterpret_cast<int64_t*>(ptrHost)[i]);
                break;
            case diopi_dtype_bool:
                printf("item %ld: %d\n", i, reinterpret_cast<bool*>(ptrHost)[i]);
                break;
            default:
                printf("unsupport dtype %s", diopiDtypeToStr(at.dtype()));
                break;
        }
    }
    printf("\n");
}

}  // namespace ascend
}  // namespace impl
