/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "aclnn.hpp"

#include <acl/acl_rt.h>

#include <functional>
#include <numeric>
#include <valarray>
#include <vector>

#include "../common/acloprunner.hpp"
#include "../common/utils.hpp"
#include "adaptor.hpp"

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
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;
    AscendTensor inAt(self1);
    if (!inAt.defined() || inAt.numel() == 0) {
        return 0;
    }
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

    aclnn("aclnnAdd", ctx, self, other, alpha, out);

    return 0;
}

int aclnnSinAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiTensorHandle_t out1) {
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    AscendTensor inAt(self1);
    if (!inAt.defined() || inAt.numel() == 0) {
        return 0;
    }
    // 创建self aclTensor
    auto ret = createAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = createAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclnn("aclnnSin", ctx, self, out);

    return 0;
}

int aclnnCosAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiTensorHandle_t out1) {
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    AscendTensor inAt(self1);
    if (!inAt.defined() || inAt.numel() == 0) {
        return 0;
    }
    // 创建self aclTensor
    auto ret = createAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = createAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclnn("aclnnCos", ctx, self, out);

    return 0;
}

int aclnnTriuAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, int64_t diagonal, diopiTensorHandle_t out1) {
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    AscendTensor inAt(self1);
    if (!inAt.defined() || inAt.numel() == 0) {
        return 0;
    }
    // 创建self aclTensor
    auto ret = createAclTensor1(self1, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = createAclTensor1(out1, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclnn("aclnnTriu", ctx, self, diagonal, out);

    return 0;
}

}  // namespace ascend
}  // namespace impl
