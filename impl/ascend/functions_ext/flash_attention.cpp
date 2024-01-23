/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/aclnn.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                                 diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal) {
#if 0
    AscendTensor qTensor(q);
    AscendTensor kTensor(k);
    AscendTensor vTensor(v);
    AscendTensor attentionOutTensor(attentionOut);
    double keepProb = 1 - pDropout;

    ASCEND_CHECK_ABORT(qTensor.dim() == 4, "The shapes of the input query should be 4-dimensional");
    ASCEND_CHECK_ABORT(kTensor.dim() == 4, "The shapes of the input key should be 4-dimensional");
    ASCEND_CHECK_ABORT(vTensor.dim() == 4, "The shapes of the input value should be 4-dimensional");
    ASCEND_CHECK_ABORT(keepProb >= 0 && keepProb <= 1, "The keep_prob value must be in range of [0, 1]");

    std::string inputLayout = "BSND";
    int64_t B = qTensor.shape[0];
    int64_t S0 = qTensor.shape[1];  // S for query
    int64_t S1 = kTensor.shape[1];  // S for key & value
    int64_t N = qTensor.shape[2];
    int64_t D = qTensor.shape[3];

    // 华为位置编码pse、paddmask直接传undefined tensor
    AscendTensor pseTensor;            // flash attention标准定义用不到位置编码
    AscendTensor paddingMaskTensor;    // 华为暂不支持，只支持定长
    AscendTensor attentionMaskTensor;  // attention mask依赖是否用到casual
    AscendTensor prefixTensor;         // 7.0.0新增接口
    if (isCausal) {
        // makeTensor(ctx, attentionMaskTensor, {S0, S1}, qTensor.type());
        // diopiScalar_t value = constructDiopiScalarT(diopi_dtype_float64, -INFINITY);
        // diopiFill(ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), &value);
        // diopiTriuInp(ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), 1);

        // 最新文档 attenMaskOptional（aclTensor*，计算输入）：数据类型支持：BOOL。数据格式支持ND。
        // 完整的attenmask矩阵（S1 * S2）
        makeTensor(ctx, attentionMaskTensor, {S0, S1}, diopi_dtype_bool);
        diopiScalar_t value = constructDiopiScalarT(diopi_dtype_int64, 1);
        diopiFill(ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), &value);
        diopiTril(
            ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()),
            1);
    }

    diopiTensorHandle_t dropMask;
    int64_t numels = B * N * S0 * S0;  // （B,N,S,S）
    dropoutGenMask(ctx, &dropMask, numels, keepProb, gen);

    diopiTensorHandle_t softmaxMaxTensor;
    diopiTensorHandle_t softmaxSumTensor;
    diopiTensorHandle_t softmaxOutTensor;  // 保留输出，暂未使用
    // // QK^T的shape: （B,N,S,S），暂时不清楚华为底层为了做优化需要保留(B,N,S,8)的用意
    // std::vector<int64_t> softmaxMaxSize{B, N, S0, 8};
    // std::vector<int64_t> softmaxSumSize{B, N, S0, 8};
    // std::vector<int64_t> softmaxSumSize{0};
    // diopiRequireTensor(ctx, &softmaxMaxTensor, &vectorToDiopiSize(softmaxMaxSize), nullptr, diopi_dtype_float32, diopi_device);
    // diopiRequireTensor(ctx, &softmaxSumTensor, &vectorToDiopiSize(softmaxSumSize), nullptr, diopi_dtype_float32, diopi_device);
    // diopiRequireTensor(ctx, &softmaxOutTensor, &vectorToDiopiSize(softmaxSumSize), nullptr, diopi_dtype_float32, diopi_device);

    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());
    int32_t innerPrecise = 0;  // 数据类型支持INT32，保留参数，暂未使用。0, fp16 high precision. 1, high performance.
    int64_t preTokens = kTensor.shape[1];
    int64_t nextTokens = 0;
    int64_t sparseMode = 0;

    aclnnFlashAttentionScoreGetWorkspaceSize(const aclTensor* qTensor,
                                             const aclTensor* kTensor,
                                             const aclTensor* vTensor,
                                             const aclTensor* pseTensor,
                                             const aclTensor* dropMask,
                                             const aclTensor* paddingMaskTensor,
                                             const aclTensor* attentionMaskTensor,
                                             const aclIntArray* prefixTensor,
                                             double softmaxScale,
                                             double keepProb,
                                             int64_t preTockens,
                                             int64_t nextTockens,
                                             int64_t N,
                                             char* inputLayoutPtr,
                                             int64_t innerPrecise,
                                             int64_t sparseMode,
                                             const aclTensor* softmaxMaxTensor,
                                             const aclTensor* softmaxSumTensor,
                                             const aclTensor* softmaxOutTensor,
                                             const aclTensor* attentionOutTensor,
                                             uint64_t* workspaceSize,
                                             aclOpExecutor** executor);

    aclnnFlashAttentionScore(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

    // *softmaxMax = softmaxMaxTensor;
    // *softmaxSum = softmaxSumTensor;
    // *softmaxOut = softmaxOutTensor;
#else
    aclnnFlashAttentionTest(ctx, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, q, k, v, pDropout, softmaxScale, isCausal);
#endif
    return diopiSuccess;
}

diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                         diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                         diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                         diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal) {
#if 0
    AscendTensor qTensor(q);
    AscendTensor kTensor(k);
    AscendTensor vTensor(v);
    AscendTensor attentionOutTensor(attentionOut);
    double keepProb = 1 - pDropout;

    ASCEND_CHECK_ABORT(qTensor.dim() == 4, "The shapes of the input query should be 4-dimensional");
    ASCEND_CHECK_ABORT(kTensor.dim() == 4, "The shapes of the input key should be 4-dimensional");
    ASCEND_CHECK_ABORT(vTensor.dim() == 4, "The shapes of the input value should be 4-dimensional");
    ASCEND_CHECK_ABORT(keepProb >= 0 && keepProb <= 1, "The keep_prob value must be in range of [0, 1]");

    std::string inputLayout = "BSND";
    int64_t B = qTensor.shape[0];
    int64_t S0 = qTensor.shape[1];  // S for query
    int64_t S1 = kTensor.shape[1];  // S for key & value
    int64_t N = qTensor.shape[2];
    int64_t D = qTensor.shape[3];

    // 华为位置编码pse、paddmask直接传undefined tensor
    AscendTensor pseTensor;            // flash attention标准定义用不到位置编码
    AscendTensor paddingMaskTensor;    // 华为暂不支持，只支持定长
    AscendTensor attentionMaskTensor;  // attention mask依赖是否用到casual
    AscendTensor prefixTensor;         // 7.0.0新增接口
    if (isCausal) {
        makeTensor(ctx, attentionMaskTensor, {S0, S1}, diopi_dtype_bool);
        diopiScalar_t value = constructDiopiScalarT(diopi_dtype_int64, 1);
        diopiFill(ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), &value);
        diopiTril(
            ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()),
            const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), 1);
    }

    diopiTensorHandle_t dropMask;
    int64_t numels = B * N * S0 * S0;  // （B,N,S,S）
    dropoutGenMask(ctx, &dropMask, numels, keepProb, gen);

    diopiTensorHandle_t gradPse;  // 保留输出，暂未使用

    char* inputLayoutPtr = const_cast<char*>(inputLayout.c_str());
    int32_t innerPrecise = 0;  // 数据类型支持INT32，保留参数，暂未使用。0, fp16 high precision. 1, high performance.
    int64_t preTokens = kTensor.shape[1];
    int64_t nextTokens = 0;
    int64_t sparseMode = 0;

    aclnnFlashAttentionScoreGradGetWorkspaceSize(const aclTensor* qTensor,
                                                 const aclTensor* kTensor,
                                                 const aclTensor* vTensor,
                                                 const aclTensor* gradOut,
                                                 const aclTensor* pseTensor,
                                                 const aclTensor* dropMask,
                                                 const aclTensor* paddingMaskTensor,
                                                 const aclTensor* attentionMaskTensor,
                                                 const aclTensor* softmaxMax,
                                                 const aclTensor* softmaxSum,
                                                 const aclTensor* softmaxOut,
                                                 const aclTensor* attentionOut,
                                                 const aclIntArray* prefixTensor,
                                                 double softmaxScale,
                                                 double keepProb,
                                                 int64_t preTokens,
                                                 int64_t nextTokens,
                                                 int64_t N,
                                                 char* inputLayoutPtr,
                                                 int64_t innerPrecise,
                                                 int64_t sparseMode,
                                                 const aclTensor* gradQ,
                                                 const aclTensor* gradK,
                                                 const aclTensor* gradV,
                                                 const aclTensor* gradPse,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

    aclnnFlashAttentionScoreGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
#else
    aclnnFlashAttentionBackwardTest(
        ctx, gradQ, gradK, gradV, gradOut, q, k, v, attentionOut, softmaxMax, softmaxSum, softmaxOut, gen, pDropout, softmaxScale, isCausal);
#endif
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
