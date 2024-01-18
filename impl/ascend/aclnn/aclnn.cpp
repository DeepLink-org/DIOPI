/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "aclnn.hpp"
#include <acl/acl_rt.h>

#include "../common/acloprunner.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

int CreateAclTensor1(diopiConstTensorHandle_t input, aclTensor** tensor) {
    impl::ascend::AscendTensor inAt(input);
    void* deviceAddr = nullptr;

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(inAt.shape().data(),
                              inAt.shape().size(),
                              getAclDataType(inAt.dtype()),
                              inAt.stride().data(),
                              0,
                              aclFormat::ACL_FORMAT_ND,
                              inAt.shape().data(),
                              inAt.shape().size(),
                              const_cast<void*>(inAt.data()));
    return ACL_SUCCESS;
}

int aclnnAddTest(int32_t deviceId, aclrtContext& context, aclrtStream& stream, diopiConstTensorHandle_t self1, diopiConstTensorHandle_t other1,
                 const diopiScalar_t* alpha1, diopiTensorHandle_t out1) {
    std::cout << "run aclnnAddTest" << std::endl;
    std::cout << "self=" << self1 << std::endl;
    std::cout << "other=" << other1 << std::endl;
    std::cout << "alpha=" << alpha1 << std::endl;
    std::cout << "out=" << out1 << std::endl;
    // 1.(固定写法)device/context/stream初始化
    // 根据自己的实际device填写deviceId
    // int32_t deviceId = 0;
    // aclrtContext context;
    // aclrtStream stream;
    // auto ret = Init(deviceId, &context, &stream);
    // // check根据自己的需要处理
    // CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2.构造输入与输出，需要根据API的接口自定义构造
    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    // 创建self aclTensor
    auto ret = CreateAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other aclTensor
    ret = CreateAclTensor1(other1, &other);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建alpha aclScalar
    auto a = getValue<float>(alpha1);
    std::cout << "scalar value a = " << a << std::endl;
    alpha = aclCreateScalar(&a, getAclDataType(alpha1->stype));
    CHECK_RET(alpha != nullptr, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAdd第一段接口
    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAdd第二段接口
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // TODO 需要将out回写
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    // AscendTensor outAt(out1);
    // void* p = nullptr;
    // diopiGetTensorData(out1, &p);
    // aclrtMemcpy(const_cast<void*>(outAt.data()), outAt.getAclMemBufferSize(), const void *src, size_t count, ACL_MEMCPY_DEVICE_TO_DEVICE)

    std::cout << "self=" << self1 << std::endl;
    std::cout << "other=" << other1 << std::endl;
    std::cout << "alpha=" << alpha1 << std::endl;
    std::cout << "out=" << out1 << std::endl;
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    // auto size = GetShapeSize(outShape);
    // std::vector<float> resultData(size, 0);
    // ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // for (int64_t i = 0; i < size; i++) {
    //     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    // }
    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    // aclDestroyTensor(self);
    // aclDestroyTensor(other);
    // aclDestroyScalar(alpha);
    // aclDestroyTensor(out);

    // 7.释放device资源，需要根据具体API的接口定义修改
    // aclrtFree(selfDeviceAddr);
    // aclrtFree(otherDeviceAddr);
    // aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    // aclrtDestroyStream(stream);
    // aclrtDestroyContext(context);
    // aclrtResetDevice(deviceId);
    // aclFinalize();
    return 0;
}

}  // namespace ascend
}  // namespace impl
