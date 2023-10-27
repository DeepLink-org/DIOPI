/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "debug.hpp"

#include "utils.hpp"

namespace impl {
namespace ascend {

void printScalar(const AscendTensor& at, void* ptrHost) {
    switch (at.dtype()) {
        case diopi_dtype_float32:
            printf("scalar: %f\n", reinterpret_cast<float*>(ptrHost)[0]);
            break;
        case diopi_dtype_float64:
            printf(" scalar: %f\n", reinterpret_cast<double*>(ptrHost)[0]);
            break;
        case diopi_dtype_int32:
            printf(" scalar: %d\n", reinterpret_cast<int*>(ptrHost)[0]);
            break;
        case diopi_dtype_int64:
            printf(" scalar: %ld\n", reinterpret_cast<int64_t*>(ptrHost)[0]);
            break;
        case diopi_dtype_bool:
            printf(" scalar: %d\n", reinterpret_cast<bool*>(ptrHost)[0]);
            break;
        default:
            printf("unsupport dtype %s", diopiDtypeToStr(at.dtype()));
            break;
    }
}

void printTensor(const AscendTensor& at, void* ptrHost) {
    size_t index = 0;
    switch (at.dtype()) {
        case diopi_dtype_float32: {
            auto ptr0 = reinterpret_cast<float*>(ptrHost);
            std::vector<float> input0(ptr0, ptr0 + at.numel());
            printVectorWithShape(input0, at.shape(), 0, index);
            return;
        }
        case diopi_dtype_float64: {
            auto ptr1 = reinterpret_cast<double*>(ptrHost);
            std::vector<double> input1(ptr1, ptr1 + at.numel());
            printVectorWithShape(input1, at.shape(), 0, index);
            return;
        }
        case diopi_dtype_int32: {
            auto ptr2 = reinterpret_cast<int32_t*>(ptrHost);
            std::vector<int32_t> input2(ptr2, ptr2 + at.numel());
            printVectorWithShape(input2, at.shape(), 0, index);
            return;
        }
        case diopi_dtype_int64: {
            auto ptr3 = reinterpret_cast<int64_t*>(ptrHost);
            std::vector<int64_t> input3(ptr3, ptr3 + at.numel());
            printVectorWithShape(input3, at.shape(), 0, index);
            return;
        }
        case diopi_dtype_bool: {
            auto ptr4 = reinterpret_cast<uint8_t*>(ptrHost);
            std::vector<uint8_t> input4(ptr4, ptr4 + at.numel());
            printVectorWithShape(input4, at.shape(), 0, index);
            return;
        }
        default:
            printf("unsupport dtype %s", diopiDtypeToStr(at.dtype()));
            return;
    }
}

void printContiguousTensor(diopiContextHandle_t ctx, const AscendTensor& at, char* name) {
    printf("==========[Tensor name %s]==========\n", name);
    if (!at.defined()) {
        printf("input tensor is nullptr.");
        return;
    }
    if (!at.isContiguous()) {
        printf("input tensor is not contiguous. break;\n");
        return;
    }
    void* ptrHost;

    printf("Tensor device: %s\n", (at.device() ? "diopi_device" : "diopi_host"));
    if (0 == at.numel() * at.elemsize()) {
        printf("tensor %s has %ld element, element size %ld.\n", name, at.numel(), at.elemsize());
        return;
    }

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
    printf("Tensor %s:\n", name);
    if (at.shape().empty()) {
        printScalar(at, ptrHost);
    } else {
        printTensor(at, ptrHost);
    }
}

}  // namespace ascend
}  // namespace impl
