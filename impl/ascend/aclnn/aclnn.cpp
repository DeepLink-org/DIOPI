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

int aclnnAddTest(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiConstTensorHandle_t other1, const diopiScalar_t* alpha1,
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

int aclnnSinTest(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiTensorHandle_t out1) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.构造输入与输出，需要根据API的接口自定义构造
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
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

int aclnnCosTest(diopiContextHandle_t ctx, diopiConstTensorHandle_t self1, diopiTensorHandle_t out1) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.构造输入与输出，需要根据API的接口自定义构造
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
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
namespace {
const size_t seedSize = sizeof(uint64_t);
const size_t offsetSize = sizeof(int64_t);

std::pair<uint64_t, int64_t> getSeedAndOffsetForMHA(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, uint64_t inc) {
    diopiTensorHandle_t stateHandle = nullptr;
    diopiGeneratorGetState(ctx, gen, &stateHandle);
    void* statePtr = nullptr;
    diopiGetTensorData(stateHandle, &statePtr);
    uint64_t currentSeedValue = 0;
    int64_t offsetValue = 0;
    memcpy(&currentSeedValue, statePtr, seedSize);
    memcpy(&offsetValue, static_cast<char*>(statePtr) + seedSize, offsetSize);
    return std::make_pair(currentSeedValue, offsetValue);
}

aclDataType dtypeConvertor(diopiDtype_t type) {
    auto dtype = getAclDataType(type);
    if (dtype == ACL_BOOL) {
        return ACL_UINT8;
    }
    return dtype;
}
}  // namespace

void dropoutGenMask(diopiContextHandle_t ctx, diopiTensorHandle_t* mask, const int64_t numels, double keepProb, diopiGeneratorHandle_t generator) {
    diopiTensorHandle_t maskTensor;
    std::vector<int64_t> maskShape{((numels + 128 - 1) / 128 * 128) / 8};
    diopiSize_t maskSize = vectorToDiopiSize(maskShape);
    diopiRequireTensor(ctx, &maskTensor, &maskSize, nullptr, diopi_dtype_bool, diopi_device);

    auto pair = getSeedAndOffsetForMHA(ctx, generator, 10);
    const int64_t seed = pair.first;
    const int64_t offset = pair.second;
    const int64_t seed1 = 0;

    std::vector<int64_t> offsetVector{0, offset};
    std::vector<int64_t> inputVector{numels};
    diopiSize_t offsetSize = vectorToDiopiSize(offsetVector);
    diopiSize_t inputSize = vectorToDiopiSize(inputVector);

    AclOpRunner<5, 1, dtypeConvertor>("StatelessDropOutGenMask", ctx)
        .addConstInput(inputSize)
        .addConstInput(keepProb, diopi_dtype_float32)
        .addConstInput(seed, diopi_dtype_int32)
        .addConstInput(seed1, diopi_dtype_int32)
        .addConstInput(offsetSize)
        .addOutput(maskTensor)
        .run();
    *mask = maskTensor;
}

int aclnnFlashAttentionTest(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t softmaxMax, diopiTensorHandle_t softmaxSum,
                            diopiTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                            diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);

    AscendTensor qTensorTmp(q);
    AscendTensor kTensorTmp(k);
    AscendTensor vTensorTmp(v);
    AscendTensor attentionMask;
    double keepProb = 1 - pDropout;

    ASCEND_CHECK_ABORT(qTensorTmp.dim() == 4, "The shapes of the input query should be 4-dimensional");
    ASCEND_CHECK_ABORT(kTensorTmp.dim() == 4, "The shapes of the input key should be 4-dimensional");
    ASCEND_CHECK_ABORT(vTensorTmp.dim() == 4, "The shapes of the input value should be 4-dimensional");
    ASCEND_CHECK_ABORT(keepProb >= 0 && keepProb <= 1, "The keep_prob value must be in range of [0, 1]");

    std::string inputLayout = "BSND";
    int64_t B = qTensorTmp.shape(0);
    int64_t S0 = qTensorTmp.shape(1);  // S for query
    int64_t S1 = kTensorTmp.shape(1);  // S for key & value
    int64_t N = qTensorTmp.shape(2);
    int64_t D = qTensorTmp.shape(3);

    aclTensor* qTensor = nullptr;
    aclTensor* kTensor = nullptr;
    aclTensor* vTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;

    aclTensor* softmaxMaxTensor = nullptr;
    aclTensor* softmaxSumTensor = nullptr;
    aclTensor* softmaxOutTensor = nullptr;

    aclTensor* pseTensor = nullptr;
    aclTensor* paddingMaskTensor = nullptr;
    aclTensor* attentionMaskTensor = nullptr;
    aclTensor* dropMaskTensor = nullptr;
    aclIntArray* prefixTensor = nullptr;

    if (isCausal) {
        // 最新文档 attenMaskOptional（aclTensor*，计算输入）：数据类型支持：BOOL。数据格式支持ND。
        // 完整的attenmask矩阵（S1 * S2）
        makeTensor(ctx, attentionMask, {S0, S1}, diopi_dtype_bool);
        diopiScalar_t value = constructDiopiScalarT(diopi_dtype_int64, 1);
        diopiFill(ctx, const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), &value);
        diopiTril(ctx, const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), 1);
    }

    diopiTensorHandle_t dropMask;
    int64_t numels = B * N * S0 * S0;  // （B,N,S,S）
    dropoutGenMask(ctx, &dropMask, numels, keepProb, gen);

    auto ret = createAclTensor1(q, &qTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(k, &kTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(v, &vTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(attentionOut, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(softmaxMax, &softmaxMaxTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(softmaxSum, &softmaxSumTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(attentionMask.tensorHandle(), &attentionMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(dropMask, &dropMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnFlashAttentionScore第一段接口
    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());
    int64_t preTokens = kTensorTmp.shape(1);
    int64_t nextTokens = 0;
    int32_t innerPrecise = 0;  // 数据类型支持INT32，保留参数，暂未使用。0, fp16 high precision. 1, high performance.
    int64_t sparseMode = 0;
    ret = aclnnFlashAttentionScoreGetWorkspaceSize(qTensor,
                                                   kTensor,
                                                   vTensor,
                                                   pseTensor,
                                                   dropMaskTensor,
                                                   paddingMaskTensor,
                                                   attentionMaskTensor,
                                                   prefixTensor,
                                                   softmaxScale,
                                                   keepProb,
                                                   preTokens,
                                                   nextTokens,
                                                   N,
                                                   inputLayoutPtr,
                                                   innerPrecise,
                                                   sparseMode,
                                                   softmaxMaxTensor,
                                                   softmaxSumTensor,
                                                   softmaxOutTensor,
                                                   attentionOutTensor,
                                                   &workspaceSize,
                                                   &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnFlashAttentionScore第二段接口
    ret = aclnnFlashAttentionScore(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScore failed. ERROR: %d\n", ret); return ret);
    // (固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

int aclnnFlashAttentionBackwardTest(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                    diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                    diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                    diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);

    AscendTensor qTensorTmp(q);
    AscendTensor kTensorTmp(k);
    AscendTensor vTensorTmp(v);
    AscendTensor attentionMask;
    double keepProb = 1 - pDropout;

    ASCEND_CHECK_ABORT(qTensorTmp.dim() == 4, "The shapes of the input query should be 4-dimensional");
    ASCEND_CHECK_ABORT(kTensorTmp.dim() == 4, "The shapes of the input key should be 4-dimensional");
    ASCEND_CHECK_ABORT(vTensorTmp.dim() == 4, "The shapes of the input value should be 4-dimensional");
    ASCEND_CHECK_ABORT(keepProb >= 0 && keepProb <= 1, "The keep_prob value must be in range of [0, 1]");

    std::string inputLayout = "BSND";
    int64_t B = qTensorTmp.shape(0);
    int64_t S0 = qTensorTmp.shape(1);  // S for query
    int64_t S1 = kTensorTmp.shape(1);  // S for key & value
    int64_t N = qTensorTmp.shape(2);
    int64_t D = qTensorTmp.shape(3);

    aclTensor* qTensor = nullptr;
    aclTensor* kTensor = nullptr;
    aclTensor* vTensor = nullptr;
    aclTensor* gradOutTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;

    aclTensor* softmaxMaxTensor = nullptr;
    aclTensor* softmaxSumTensor = nullptr;
    aclTensor* softmaxOutTensor = nullptr;

    aclTensor* pseTensor = nullptr;
    aclTensor* paddingMaskTensor = nullptr;
    aclTensor* attentionMaskTensor = nullptr;
    aclTensor* dropMaskTensor = nullptr;
    aclIntArray* prefixTensor = nullptr;

    aclTensor* gradQTensor = nullptr;
    aclTensor* gradKTensor = nullptr;
    aclTensor* gradVTensor = nullptr;
    aclTensor* gradPseTensor = nullptr;

    if (isCausal) {
        // 最新文档 attenMaskOptional（aclTensor*，计算输入）：数据类型支持：BOOL。数据格式支持ND。
        // 完整的attenmask矩阵（S1 * S2）
        makeTensor(ctx, attentionMask, {S0, S1}, diopi_dtype_bool);
        diopiScalar_t value = constructDiopiScalarT(diopi_dtype_int64, 1);
        diopiFill(ctx, const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), &value);
        diopiTril(ctx, const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), 1);
    }

    diopiTensorHandle_t dropMask;
    int64_t numels = B * N * S0 * S0;  // （B,N,S,S）
    dropoutGenMask(ctx, &dropMask, numels, keepProb, gen);

    auto ret = createAclTensor1(q, &qTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(k, &kTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(v, &vTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(gradOut, &gradOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(attentionOut, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(softmaxMax, &softmaxMaxTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(softmaxSum, &softmaxSumTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(attentionMask.tensorHandle(), &attentionMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(dropMask, &dropMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(gradQ, &gradQTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(gradK, &gradKTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(gradV, &gradVTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnFlashAttentionScoreGradGetWorkspaceSize第一段接口
    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());
    int64_t preTokens = kTensorTmp.shape(1);
    int64_t nextTokens = 0;
    int32_t innerPrecise = 0;  // 数据类型支持INT32，保留参数，暂未使用。0, fp16 high precision. 1, high performance.
    int64_t sparseMode = 0;
    ret = aclnnFlashAttentionScoreGradGetWorkspaceSize(qTensor,
                                                       kTensor,
                                                       vTensor,
                                                       gradOutTensor,
                                                       pseTensor,
                                                       dropMaskTensor,
                                                       paddingMaskTensor,
                                                       attentionMaskTensor,
                                                       softmaxMaxTensor,
                                                       softmaxSumTensor,
                                                       softmaxOutTensor,
                                                       attentionOutTensor,
                                                       prefixTensor,
                                                       softmaxScale,
                                                       keepProb,
                                                       preTokens,
                                                       nextTokens,
                                                       N,
                                                       inputLayoutPtr,
                                                       innerPrecise,
                                                       sparseMode,
                                                       gradQTensor,
                                                       gradKTensor,
                                                       gradVTensor,
                                                       gradPseTensor,
                                                       &workspaceSize,
                                                       &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnFlashAttentionScoreGrad第二段接口
    ret = aclnnFlashAttentionScoreGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGrad failed. ERROR: %d\n", ret); return ret);
    // (固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

}  // namespace ascend
}  // namespace impl
