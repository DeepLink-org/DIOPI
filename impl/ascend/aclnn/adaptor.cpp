

/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "adaptor.hpp"

#include <numeric>

namespace impl {
namespace ascend {
int createAclTensor1(diopiConstTensorHandle_t input, aclTensor** tensor) {
    impl::ascend::AscendTensor inAt(input);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(inAt.getAclMemShape().data(),
                              inAt.getAclMemShape().size(),
                              inAt.getAclDataType(),
                              inAt.stride().data(),
                              0,
                              inAt.getAclDataFormat(),
                              inAt.getAclMemShape().data(),
                              inAt.getAclMemShape().size(),
                              const_cast<void*>(inAt.data()));
    return ACL_SUCCESS;
}

// void printContiTensor(const aclTensor& tensor, const void* tensorPtr) {
//     int64_t* shape = nullptr;
//     uint64_t num = 0;
//     aclGetViewShape(&tensor, &shape, &num);
//     std::vector<int64_t> shapeVec(shape, shape + num);
//     int64_t size = std::accumulate(shapeVec.begin(), shapeVec.end(), 1, std::multiplies<>());
//     std::cout << "acl tensor size = " << size << std::endl;
//     std::vector<float> result(size, 0);
//     auto ret = aclrtMemcpy(result.data(), result.size() * sizeof(result[0]), tensorPtr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
//     if (ret != ACL_SUCCESS) {
//         std::cout << "copy result from device to host failed. ERROR: " << ret << std::endl;
//         return;
//     }
//     for (int64_t i = 0; i < size; i++) {
//         std::cout << "printContiTensor result[" << i << "] is: " << result[i] << std::endl;
//     }
// }

// void printContiTensor(const aclTensor& tensor, diopiConstTensorHandle_t diopi) {
//     const void* p = nullptr;
//     diopiGetTensorDataConst(diopi, &p);
//     return printContiTensor(tensor, p);
// }


}  // namespace ascend
}  // namespace impl
