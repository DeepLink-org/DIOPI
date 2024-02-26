/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/aclnn.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

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
    // // update offset
    // inc = ((inc + 3) / 4) * 4;
    // uint64_t updateOffset = offsetValue + inc;
    // memcpy(static_cast<char*>(statePtr) + seedSize, &updateOffset, offsetSize);
    // diopiGeneratorSetState(gen, stateHandle);
    return std::make_pair(currentSeedValue, offsetValue);
}

aclDataType dtypeConvertor(diopiDtype_t type) {
    auto dtype = getAclDataType(type);
    if (dtype == ACL_BOOL) {
        return ACL_UINT8;
    }
    return dtype;
}

void dropoutGenMask(diopiContextHandle_t ctx, diopiTensorHandle_t* mask, const int64_t numels, const double keepProb, diopiGeneratorHandle_t generator) {
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
}  // namespace

int aclnnFlashAttentionAdaptor(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                               diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
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
    // ASCEND_CHECK_ABORT(S0 % 16 == 0, "S must be a multiple of 16");
    // ASCEND_CHECK_ABORT(S1 % 16 == 0, "S must be a multiple of 16");
    // ASCEND_CHECK_ABORT(D == 64 || D == 96 || D == 128 || D == 256, "D must be 64, 96, 128, 256");

    diopiTensorHandle_t softmaxMaxTensor;
    diopiTensorHandle_t softmaxSumTensor;
    diopiTensorHandle_t softmaxOutTensor;  // 保留输出，暂未使用
    // QK^T的shape: （B,N,S,S），暂时不清楚华为底层为了做优化需要保留(B,N,S,8)的用意
    // 32字节对齐
    std::vector<int64_t> softmaxMaxShape{B, N, S0, 8};
    std::vector<int64_t> softmaxSumShape{B, N, S0, 8};
    std::vector<int64_t> softmaxOutShape{0};
    diopiSize_t softmaxMaxSize = vectorToDiopiSize(softmaxMaxShape);
    diopiSize_t softmaxSumSize = vectorToDiopiSize(softmaxSumShape);
    diopiSize_t softmaxOutSize = vectorToDiopiSize(softmaxOutShape);
    diopiRequireTensor(ctx, &softmaxMaxTensor, &softmaxMaxSize, nullptr, diopi_dtype_float32, diopi_device);
    diopiRequireTensor(ctx, &softmaxSumTensor, &softmaxSumSize, nullptr, diopi_dtype_float32, diopi_device);
    diopiRequireTensor(ctx, &softmaxOutTensor, &softmaxOutSize, nullptr, diopi_dtype_float32, diopi_device);

    aclTensor* qTensor = nullptr;
    aclTensor* kTensor = nullptr;
    aclTensor* vTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;

    aclTensor* softmaxMaxAclTensor = nullptr;
    aclTensor* softmaxSumAclTensor = nullptr;
    aclTensor* softmaxOutAclTensor = nullptr;

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
        diopiTril(ctx, const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), const_cast<diopiTensorHandle_t>(attentionMask.tensorHandle()), 0);
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
    ret = createAclTensor1(softmaxMaxTensor, &softmaxMaxAclTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(softmaxSumTensor, &softmaxSumAclTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = createAclTensor1(softmaxOutTensor, &softmaxOutAclTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (isCausal) {
        ret = createAclTensor1(attentionMask.tensorHandle(), &attentionMaskTensor);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = createAclTensor1(dropMask, &dropMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
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
                                                   softmaxMaxAclTensor,
                                                   softmaxSumAclTensor,
                                                   softmaxOutAclTensor,
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
    *softmaxMax = softmaxMaxTensor;
    *softmaxSum = softmaxSumTensor;
    *softmaxOut = softmaxOutTensor;
    return 0;
}

int aclnnFlashAttentionBackwardAdaptor(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
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
    ASCEND_CHECK_ABORT(S0 % 16 == 0, "S must be a multiple of 16");
    ASCEND_CHECK_ABORT(S1 % 16 == 0, "S must be a multiple of 16");
    ASCEND_CHECK_ABORT(D == 64 || D == 96 || D == 128 || D == 256, "D must be 64, 96, 128, 256");

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

diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                                 diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal) {
    aclnnFlashAttentionAdaptor(ctx, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, q, k, v, pDropout, softmaxScale, isCausal);
    return diopiSuccess;
}

diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                         diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                         diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                         diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal) {
    aclnnFlashAttentionBackwardAdaptor(
        ctx, gradQ, gradK, gradV, gradOut, q, k, v, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, pDropout, softmaxScale, isCausal);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
