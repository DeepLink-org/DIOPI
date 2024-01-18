/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

void dropoutGenMask(diopiContextHandle_t ctx, diopiTensorHandle_t* mask, const int64_t numels, double keepProb, diopiGeneratorHandle_t generator) {
    diopiTensorHandle_t maskTensor;
    std::vector<int64_t> maskShape{((numels + 128 - 1) / 128 * 128) / 8};
    diopiSize_t maskSize = vectorToDiopiSize(maskShape);
    diopiRequireTensor(ctx, &maskTensor, &maskSize, nullptr, diopi_dtype_bool, diopi_device);

    auto pair = getSeedAndOffset(ctx, generator, 10);
    const int64_t seed = pair.first;
    const int64_t offset = pair.second;
    const int64_t seed1 = 0;

    diopiSize_t offsetSize = vectorToDiopiSize(std::vector<int64_t>{0, offset});
    diopiSize_t inputSize = vectorToDiopiSize(std::vector<int64_t>{numels});

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

/**
 * @brief Compute the forward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] q Query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] k Key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] v Value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float32, float16, float64].
 * @param[in] p_dropout The probability of dropout op.
 * @param[in] softmax_scale The temperature to use for the softmax attention. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] is_causal Whether to apply causal attention mask.
 * @param[out] attention_out Tensor containing the result after applying flash attention. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float16,
 * float32, float64].
 * @param[out] softmax_max Tensor representing the maximum of the softmax values.
 * @param[out] softmax_sum Tensor representing the sum of the softmax values.
 * @param[out] softmax_out Tensor representing the output of the softmax attention.
 * @param[out] gen Handle for the random number generator used in dropout op.
 */
diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                                 diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                 diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal) {
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
            ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), 1);
    }

    diopiTensorHandle_t dropMask;
    int64_t numels = B * N * S0 * S0;  // （B,N,S,S）
    dropoutGenMask(ctx, &dropMask, numels, keepProb, gen);

    diopiTensorHandle_t softmaxMaxTensor;
    diopiTensorHandle_t softmaxSumTensor;
    diopiTensorHandle_t softmaxOutTensor;  // 保留输出，暂未使用
    // QK^T的shape: （B,N,S,S），暂时不清楚华为底层为了做优化需要保留(B,N,S,8)的用意
    std::vector<int64_t> softmaxMaxSize{B, N, S0, 8};
    std::vector<int64_t> softmaxSumSize{B, N, S0, 8};
    std::vector<int64_t> softmaxSumSize{0};
    diopiRequireTensor(ctx, &softmaxMaxTensor, &vectorToDiopiSize(softmaxMaxSize), nullptr, diopi_dtype_float32, diopi_device);
    diopiRequireTensor(ctx, &softmaxSumTensor, &vectorToDiopiSize(softmaxSumSize), nullptr, diopi_dtype_float32, diopi_device);
    diopiRequireTensor(ctx, &softmaxOutTensor, &vectorToDiopiSize(softmaxSumSize), nullptr, diopi_dtype_float32, diopi_device);

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

    *softmaxMax = softmaxMaxTensor;
    *softmaxSum = softmaxSumTensor;
    *softmaxOut = softmaxOutTensor;
    return diopiSuccess;
}

/**
 * @brief Compute the backward pass for Flash Attention.
 * @param[in] ctx The diopi context.
 * @param[in] grad_out The gradient of the output tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[in] q Query tensor from the forward pass. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[in] k Key tensor from the forward pass. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[in] v Value tensor from the forward pass. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[in] attention_out Output tensor from the forward pass. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[in] softmax_max Tensor representing the maximum of the softmax values from the forward pass. shape = [batch_size, head_num, q_seq_len]. type =
 * [float32].
 * @param[in] softmax_sum Tensor representing the sum of the softmax values from the forward pass. shape = [batch_size, head_num, q_seq_len]. type = [float32].
 * @param[in] softmax_out Tensor representing the output of the softmax attention from the forward pass. shape = [batch_size, head_num, q_seq_len]. type =
 * [float32].
 * @param[in] gen Handle representing the random number generator used for dropout in the forward pass.
 * @param[in] p_dropout The probability of dropout op.
 * @param[in] softmax_scale The temperature to use for the softmax attention. By default, softmax\_scale=\frac{1}{\sqrt{d_k}}
 * @param[in] is_causal Whether to apply causal attention mask.
 * @param[out] grad_q The gradient of the query tensor. shape = [batch_size, q_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[out] grad_k The gradient of the key tensor. shape = [batch_size, k_seq_len, head_num, head_dim]. type = [float16, float32].
 * @param[out] grad_v The gradient of the value tensor. shape = [batch_size, v_seq_len, head_num, head_dim]. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                                   diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                                   diopiConstTensorHandle_t v, diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax,
                                                   diopiConstTensorHandle_t softmaxSum, diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen,
                                                   double pDropout, double softmaxScale, bool isCausal) {
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
            ctx, const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), const_cast<diopiTensorHandle_t>(attentionMaskTensor.tensorHandle()), 1);
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
                                                 int64_t preTockens,
                                                 int64_t nextTockens,
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
}

}  // namespace ascend
}  // namespace impl
