/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "aclnn.hpp"

#include <acl/acl_rt.h>

#include <functional>
#include <numeric>
#include <valarray>
#include <vector>

#include "../common/acloprunner.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

int createAclTensor1(diopiConstTensorHandle_t input, aclTensor** tensor) {
    impl::ascend::AscendTensor inAt(input);
    void* deviceAddr = nullptr;

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

aclScalar* createAclScalar1(const diopiScalar_t* input) {
    // 创建alpha aclScalar
    if (input->stype == diopiDtype_t::diopi_dtype_float64) {
        auto v = getValue<double>(input);
        return aclCreateScalar(&v, getAclDataType(input->stype));
    } else {
        auto v = getValue<int64_t>(input);
        return aclCreateScalar(&v, getAclDataType(input->stype));
    }
    return nullptr;
}

void printContiguousTensor(const aclTensor& tensor, const void* tensorPtr) {
    int64_t* shape = nullptr;
    uint64_t num = 0;
    aclGetViewShape(&tensor, &shape, &num);
    std::vector<int64_t> shapeVec(shape, shape + num);
    int64_t size = std::accumulate(shapeVec.begin(), shapeVec.end(), 1, std::multiplies<>());
    std::vector<float> result(size, 0);
    auto ret = aclrtMemcpy(result.data(), result.size() * sizeof(result[0]), tensorPtr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return;);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, result[i]);
    }
}

void printContiguousTensor(const aclTensor& tensor, diopiConstTensorHandle_t diopi) {
    const void* p = nullptr;
    diopiGetTensorDataConst(diopi, &p);
    return printContiguousTensor(tensor, p);
}

int aclnnAddAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiConstTensorHandle_t other1, const diopiScalar_t* alpha1,
                    diopiTensorHandle_t out1) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.构造输入与输出，需要根据API的接口自定义构造
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    // 创建self aclTensor
    auto ret = createAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other aclTensor
    ret = createAclTensor1(other1, &other);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建alpha aclScalar
    alpha = createAclScalar1(alpha1);

    CHECK_RET(alpha != nullptr, return ret);
    // 创建out aclTensor
    ret = createAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 2.调用CANN算子库API
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
    // 3.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

int aclnnSinAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiTensorHandle_t out1) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.构造输入与输出，需要根据API的接口自定义构造
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    AscendTensor inAt(self1);
    if (inAt.numel() == 0) {
        return 0;
    }
    // 创建self aclTensor
    auto ret = createAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = createAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 2.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnSin第一段接口
    ret = aclnnSinGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnSin第二段接口
    ret = aclnnSin(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSin failed. ERROR: %d\n", ret); return ret);
    // 3.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

int aclnnCosAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiTensorHandle_t out1) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.构造输入与输出，需要根据API的接口自定义构造
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    AscendTensor inAt(self1);
    if (inAt.numel() == 0) {
        return 0;
    }
    // 创建self aclTensor
    auto ret = createAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = createAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 2.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCos第一段接口
    ret = aclnnCosGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCosGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnCos第二段接口
    ret = aclnnCos(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCos failed. ERROR: %d\n", ret); return ret);
    // 3.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

}  // namespace ascend
}  // namespace impl
