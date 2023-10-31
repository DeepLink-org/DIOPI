/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include "cuda.h"
#include "context.h"
#include "helper.hpp"

#define checkCudaErrors(x)                                                                                            \
    {                                                                                                                 \
        CUresult err = x;                                                                                             \
        if (err != CUDA_SUCCESS) {                                                                                    \
            std::cout << "checkCudaErrors:" << __FILE__ << ":" << __LINE__ << ": " << #x << ": " << err << std::endl; \
        }                                                                                                             \
    }

namespace {

    unsigned int next_power_of_2(unsigned int x) {
        return std::pow(2, std::ceil(std::log(x) / std::log(2)));
    }

    class CuKernelModuleLoader {
        private:
            CUmodule cudaModule_;
            CUfunction  function_;

        public:
            CuKernelModuleLoader(const char* function_name) {
                std::string fatbin_path = std::getenv("HOME") + std::string("/.triton/diopi/") + function_name + std::string(".fatbin");

                checkCudaErrors(cuModuleLoad(&cudaModule_, fatbin_path.c_str()));
                //checkCudaErrors(cuModuleLoadFatBinary(&cudaModule_, cuModuleBin_.data_ptr()));
                checkCudaErrors(cuModuleGetFunction(&function_, cudaModule_, function_name));
            }

            ~CuKernelModuleLoader() {
                checkCudaErrors(cuModuleUnload(cudaModule_));
            }
        CUfunction& get() {
            return function_;
        }
    };

    template<int kernel_id>
    class TritonKernelRunner {
        public:

        void run(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, const char* kernel_name, CUstream stream, void** kernelParams, void** extra) {
            static CuKernelModuleLoader loader(kernel_name);
            if (std::getenv("DIOPI_DEBUG_TRITON") != nullptr) {
                std::cout << kernel_name << ":gridX:" << gridX << ",gridY:" << gridY << ",gridZ:" << gridZ << ",num_warps:" << num_warps << ",num_ctas:" << num_ctas << ",cluster_x:" << clusterDimX << ",cluster_y:" << clusterDimY << ",cluster_z:" << clusterDimZ << ",shared_memory:" << shared_memory << std::endl;
            }
            if (gridX * gridY * gridZ > 0) {
                if (num_ctas == 1) {
                    checkCudaErrors(cuLaunchKernel(loader.get(), gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, kernelParams, extra));
                } else {
                    /*
                    CUlaunchAttribute launchAttr[2];
                    launchAttr[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
                    launchAttr[0].value.clusterDim.x = clusterDimX;
                    launchAttr[0].value.clusterDim.y = clusterDimY;
                    launchAttr[0].value.clusterDim.z = clusterDimZ;
                    launchAttr[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
                    launchAttr[1].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
                    CUlaunchConfig config;
                    config.gridDimX = gridX * clusterDimX;
                    config.gridDimY = gridY * clusterDimY;
                    config.gridDimZ = gridZ * clusterDimZ;
                    config.blockDimX = 32 * num_warps;
                    config.blockDimY = 1;
                    config.blockDimZ = 1;
                    config.sharedMemBytes = shared_memory;
                    config.hStream = stream;
                        config.attrs = launchAttr;
                        config.numAttrs = 2;
                    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
                    if (cuLaunchKernelExHandle == NULL) {{
                        cuLaunchKernelExHandle = getLaunchKernelExHandle();
                    }}
                    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
                    */
                    int gridDimX = gridX * clusterDimX;
                    int gridDimY = gridY * clusterDimY;
                    int gridDimZ = gridZ * clusterDimZ;

                    int blockDimX = 32 * num_warps;
                    int blockDimY = 1;
                    int blockDimZ = 1;

                    checkCudaErrors(cuLaunchKernel(loader.get(), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, shared_memory, stream, kernelParams, extra));
                }
            }
        }
    };

    #define TritonKernelRunner_t TritonKernelRunner<__LINE__>

} // namespace

extern "C" {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    /*
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv[grid](
        K, DestLoc, Out,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    */
    impl::aten::setCurCtx(ctx);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atDestLoc = impl::aten::buildATen(destLoc);

    unsigned int seq_len = atDestLoc.numel();
    unsigned int head_num = atK.size(1);
    unsigned int head_dim = atK.size(2);
    unsigned int BLOCK_HEAD = next_power_of_2(head_num);

    assert(atK.size(1) == atOut.size(1) && atK.size(2) == atOut.size(2));

    TritonKernelRunner_t kernel;
    const int gridX = seq_len;
    const int gridY = 1;
    const int gridZ = 1;
    const int num_warps = 1;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 0;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    CUstream stream = static_cast<CUstream>(stream_handle);
    const char* kernel_name = "_fwd_kernel_destindex_copy_kv_0d1d2d3de4de5c6de7de8c9de";
    void* k_ptr = atK.data_ptr();
    void* out_ptr = atOut.data_ptr();
    void* destLoc_ptr = atDestLoc.data_ptr();
    unsigned int k_stride[] = {atK.stride(0), atK.stride(1), atK.stride(2)};
    unsigned int out_stride[] = {atOut.stride(0), atOut.stride(1), atOut.stride(2)};
    void** extra_param = NULL;
    void *kernel_params[] = {
        &k_ptr, &destLoc_ptr, &out_ptr,
        k_stride, k_stride + 1,
        out_stride, out_stride + 1,
        &head_num,
      };
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    //cuCtxSynchronize();
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atQ = impl::aten::buildATen(q);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atBLoc = impl::aten::buildATen(bLoc);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqLen);
    at::Tensor atOut = impl::aten::buildATen(attentionOut);

    int block = 32;
    int dimK = atK.size(-1);
    float smScale = 1.0 / std::sqrt(dimK);  // 计算scale系数
    int batch = atBLoc.size(0);
    int head_num = atQ.size(1);

    TritonKernelRunner_t kernel;
    const int gridX = batch;
    const int gridY = head_num;
    const int gridZ = std::max(1, maxInputLen / block);
    const int num_warps = 4;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 256;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    const char* kernel_name = "_fwd_kernel_token_atttention_0d1d23d4d5d67d89c10de11de12c13de14de15c1617c";
    CUstream stream = static_cast<cudaStream_t>(stream_handle);
    void* q_ptr = atQ.data_ptr();
    void* k_ptr = atK.data_ptr();
    void* b_loc_ptr = atBLoc.data_ptr();
    void* b_start_loc_ptr = atBStartLoc.data_ptr();
    void* b_seq_len_ptr = atBSeqLen.data_ptr();
    void* out_ptr = atOut.data_ptr();

    unsigned int q_stride[] = {atQ.stride(0), atQ.stride(1), atQ.stride(2)};
    unsigned int k_stride[] = {atK.stride(0), atK.stride(1), atK.stride(2)};
    unsigned int b_loc_stride[] = {atBLoc.stride(0), atBLoc.stride(1)};
    unsigned int out_stride[] = {atOut.stride(0), atOut.stride(1)};

    void** extra_param = NULL;
    void* kernel_params[] = {
        &q_ptr,
        &k_ptr,
        &smScale,
        &b_loc_ptr,
        &b_start_loc_ptr,
        &b_seq_len_ptr,
        &maxInputLen,
        &out_ptr,
        b_loc_stride,
        q_stride,
        q_stride + 1,
        k_stride,
        k_stride + 1,
        out_stride,
    };
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqlen,
                                               int maxInputLen, int otherKVIndex) {
    const char* kernel_name = "_fwd_kernel_token_softmax_reducev_0d1d2d3d4d5d678c9de10de11c12de13de14c1516c17";
    const int BLOCK = 64;
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atV = impl::aten::buildATen(v);
    at::Tensor atLogics = impl::aten::buildATen(logics);
    at::Tensor atBLoc = impl::aten::buildATen(bLoc);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqlen = impl::aten::buildATen(bSeqlen);
    const int batch = atBSeqlen.size(0);
    const int head = atLogics.size(0);
    void* logics_ptr = atLogics.data_ptr();
    void* out_ptr = atOut.data_ptr();
    void* v_ptr = atV.data_ptr();
    void* bLoc_ptr = atBLoc.data_ptr();
    void* bStartLoc_ptr = atBStartLoc.data_ptr();
    void* bSeqLen_ptr = atBSeqlen.data_ptr();
    unsigned int logics_stride[] = {atLogics.stride(0), atLogics.stride(1)};
    unsigned int v_stride[] = {atV.stride(0), atV.stride(1)};
    unsigned int o_stride[] = {atOut.stride(0), atOut.stride(1)};
    unsigned int b_loc_stride[] = {atBLoc.stride(0), atBLoc.stride(1)};
    void** extra_param = NULL;
    void* kernel_params[] = {
        &logics_ptr, &v_ptr, &out_ptr, &bLoc_ptr, &bStartLoc_ptr, &bSeqLen_ptr, &maxInputLen,
        logics_stride,
        v_stride, v_stride + 1,
        o_stride, o_stride + 1,
        b_loc_stride,
        &otherKVIndex
    };

    const int gridX = batch;
    const int gridY = head;
    const int gridZ = 1;
    const int num_warps = 1;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 256;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    CUstream stream = static_cast<CUstream>(stream_handle);
    TritonKernelRunner_t kernel;
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);

    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqlen, int maxInputLen) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atQ = impl::aten::buildATen(q);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atV = impl::aten::buildATen(v);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqlen);
    at::Tensor atOut = impl::aten::buildATen(out);

    const int Lq = atQ.size(-1);
    int Lk = atK.size(-1);
    const int Lv = atV.size(-1);
    assert(Lq == Lk && Lk == Lv);
    assert(Lk == 16 || Lk == 32 || Lk == 64 || Lk == 128);

    int block = 64;
    float smScale = 1.0 / std::sqrt(Lq);  // 计算scale系数
    const int batch = atBSeqLen.size(0);
    const int head = atQ.size(1);

    const int gridX = batch;
    const int gridY = head;
    const int gridZ = std::max(1, maxInputLen / block);
    const int num_warps = Lk <= 64 ? 4 : 8;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 40960;


    const char* kernel_name = "context_attention_fwd_kernel_0d1d2d34d5d6d7de8de9c10de11de12c13de14de15c16de17de18c";

    void* q_ptr = atQ.data_ptr();
    void* k_ptr = atK.data_ptr();
    void* v_ptr = atV.data_ptr();
    void* b_start_loc_ptr = atBStartLoc.data_ptr();
    void* b_seq_len_ptr = atBSeqLen.data_ptr();
    void* out_ptr = atOut.data_ptr();

    unsigned int q_stride[] = {atQ.stride(0), atQ.stride(1), atQ.stride(2)};
    unsigned int k_stride[] = {atK.stride(0), atK.stride(1), atK.stride(2)};
    unsigned int v_stride[] = {atV.stride(0), atV.stride(1), atV.stride(2)};
    unsigned int out_stride[] = {atOut.stride(0), atOut.stride(1), atOut.stride(2)};
    void** extra_param = NULL;
    void* kernel_params[] = {
        &q_ptr, &k_ptr, &v_ptr, &smScale,
        &b_start_loc_ptr, &b_seq_len_ptr, &out_ptr,
        &q_stride[0], &q_stride[1],
        &k_stride[0], &k_stride[1],
        &v_stride[0], &v_stride[1],
        &out_stride[0], &out_stride[1],
        &block, &Lk, &block,
    };
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    CUstream stream = static_cast<CUstream>(stream_handle);
    TritonKernelRunner_t kernel;
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiApplyPenalty(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t presence_penalty,
                                         diopiConstTensorHandle_t frequency_penalty, diopiConstTensorHandle_t p_token_ids,
                                         diopiConstTensorHandle_t p_token_counts, diopiConstTensorHandle_t p_cumsum_seq_len, int p_max_len_in_batch) {
       /*
    assert Logits.is_contiguous()
    BLOCK = triton.next_power_of_2(p_max_len_in_batch)
    if BLOCK <= 512:
        BLOCK = 512
    elif BLOCK <= 1024:
        BLOCK = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0], )](
        Logits, presence_penalty, frequency_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK
    )
    */
    impl::aten::setCurCtx(ctx);
    at::Tensor atLogits = impl::aten::buildATen(logits);
    assert(atLogits.is_contiguous());
    at::Tensor atPresencePenalty = impl::aten::buildATen(presence_penalty);
    at::Tensor atFrequencyPenalty = impl::aten::buildATen(frequency_penalty);
    at::Tensor atPTokenIds = impl::aten::buildATen(p_token_ids);
    at::Tensor atPTokenCounts = impl::aten::buildATen(p_token_counts);
    at::Tensor atPCumsumSeqLen = impl::aten::buildATen(p_cumsum_seq_len);
    int BLOCK = next_power_of_2(p_max_len_in_batch);
    if (BLOCK <= 512) {
        BLOCK = 512;
    } else {
        if (BLOCK <= 1024) {
            BLOCK = 1024;
        }
    }

    TritonKernelRunner_t kernel;
    const int gridX = atLogits.size(0);
    const int gridY = 1;
    const int gridZ = 1;
    int num_warps = 8;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 0;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    const char* kernel_name = "_fwd_kernel_apply_penalty_0d1d2d3d4d5d67c";
    CUstream stream = static_cast<cudaStream_t>(stream_handle);
    void* logits_ptr = atLogits.data_ptr();
    void* presence_penalty_ptr = atPresencePenalty.data_ptr();
    void* frequency_penalty_ptr = atFrequencyPenalty.data_ptr();
    void* p_token_ids_ptr = atPTokenIds.data_ptr();
    void* p_token_counts_ptr = atPTokenCounts.data_ptr();
    void* p_cumsum_seq_len_ptr = atPCumsumSeqLen.data_ptr();
    int logits_stride[] = {atLogits.stride(0), atLogits.stride(1)};

    void** extra_param = NULL;
    void* kernel_params[] = {&logits_ptr,
                             &presence_penalty_ptr,
                             &frequency_penalty_ptr,
                             &p_token_ids_ptr,
                             &p_token_counts_ptr,
                             &p_cumsum_seq_len_ptr,
                             &logits_stride[0],
                             &logits_stride[1],
                             &num_warps,
                             &BLOCK};
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;

}


}  // extern "C"
