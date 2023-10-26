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

#include "context.h"
#include "cuda.h"
#include "helper.hpp"

#define checkCudaErrors(x)                                                                                    \
    {                                                                                                         \
        CUresult err = x;                                                                                     \
        if (err != CUDA_SUCCESS) {                                                                            \
            std::cout << "checkCudaErrors:" << __FILE__ << ":" << __LINE__ << #x << ": " << err << std::endl; \
        }                                                                                                     \
    }

namespace {

bool read_file(std::string filename, std::vector<char>& buffer) {
    std::ifstream infile(filename.c_str(), std::ifstream::binary);
    if (!infile.is_open()) {
        printf("Read File:%s Error ... \n", filename.c_str());
        return false;
    }

    // 获取文件大小
    infile.seekg(0, std::ifstream::end);
    long size = infile.tellg();
    infile.seekg(0);

    buffer.resize(size);
    // read content of infile
    infile.read(&buffer[0], size);
    infile.close();
    return true;
}

class CuKernelModuleLoader {
private:
    CUmodule cudaModule_;
    CUfunction function_;
    std::vector<char> buffer;

public:
    CuKernelModuleLoader(const char* function_name) {
        std::string fatbin_path = std::getenv("HOME") + std::string("/.triton/diopi/") + function_name + std::string(".fatbin");
        read_file(fatbin_path, buffer);
        if (buffer.size() <= 0) {
            std::cout << "load " << fatbin_path << " failed" << std::endl;
        }
        checkCudaErrors(cuModuleLoadFatBinary(&cudaModule_, buffer.data()));
        checkCudaErrors(cuModuleGetFunction(&function_, cudaModule_, function_name));
    }

    ~CuKernelModuleLoader() { checkCudaErrors(cuModuleUnload(cudaModule_)); }
    CUfunction& get() { return function_; }
};

template <int kernel_id>
class TritonKernelRunner {
public:
    void run(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory,
             const char* kernel_name, CUstream stream, void** kernelParams, void** extra) {
        static CuKernelModuleLoader loader(kernel_name);
        if (gridX * gridY * gridZ > 0) {
            if (num_ctas == 1) {
                checkCudaErrors(cuLaunchKernel(loader.get(), gridX, gridY, gridZ, 32 * num_warps, 1, 1, shared_memory, stream, kernelParams, extra));
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

                checkCudaErrors(
                    cuLaunchKernel(loader.get(), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, shared_memory, stream, kernelParams, extra));
            }
        }
    }
};

#define TritonKernelRunner_t TritonKernelRunner<__LINE__>

unsigned int next_power_of_2(unsigned int x) { return std::pow(2, std::ceil(std::log(x) / std::log(2))); }

}  // namespace

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

    int seq_len = atK.size(0);
    int head_num = atK.size(1);
    int head_dim = atK.size(2);
    int BLOCK_HEAD = next_power_of_2(head_num);

    assert(atK.size(1) == atOut.size(1) && atK.size(2) == atOut.size(2));

    TritonKernelRunner_t kernel;
    const int gridX = atDestLoc.size(0);
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
    const char* kernel_name = "_fwd_kernel_destindex_copy_kv_0d1d2d3c4c5c6c7c8c9c";
    CUstream stream = static_cast<cudaStream_t>(stream_handle);
    void* k_ptr = atK.data_ptr();
    void* out_ptr = atOut.data_ptr();
    void* destLoc_ptr = atDestLoc.data_ptr();
    unsigned int k_stride[] = {atK.stride(0), atK.stride(1), atK.stride(2)};
    unsigned int out_stride[] = {atOut.stride(0), atOut.stride(1), atOut.stride(2)};
    void** extra_param = NULL;
    void* kernel_params[] = {&k_ptr, &destLoc_ptr, &out_ptr, &k_stride[0], &k_stride[1], &k_stride[2], &out_stride[0], &out_stride[1], &out_stride[2]};
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
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
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqlen);
    at::Tensor atOut = impl::aten::buildATen(attentionOut);

    int block = 32;
    int dimK = atK.size(2);
    float smScale = 1.0 / std::sqrt(dimK);  // 计算scale系数
    int batch = atBLoc.size(0);
    int head_num = atQ.size(1);

    TritonKernelRunner_t kernel;
    const int gridX = batch;
    const int gridY = head_num;
    const int gridZ = maxInputLen / block;
    const int num_warps = 4;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 0;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    const char* kernel_name = "_fwd_kernel_token_att1_0d1d23d4d5d6de7d89c10de11de12c13de14de15c16de17c";
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
    void* kernel_params[] = {&q_ptr,
                             &k_ptr,
                             smScale,
                             &b_loc_ptr,
                             &b_start_loc_ptr,
                             &b_seq_len_ptr,
                             maxInputLen,
                             &out_ptr,
                             &b_loc_stride[0],
                             &b_loc_stride[1],
                             &q_stride[0],
                             &q_stride[1],
                             &q_stride[2],
                             &k_stride[0],
                             &k_stride[1],
                             &k_stride[2],
                             &out_stride[0],
                             &out_stride[1]};
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;

    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                               int maxInputLen, int otherKVIndex) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atLogics = impl::aten::buildATen(logics);
    at::Tensor atV = impl::aten::buildATen(v);
    at::Tensor atBLoc = impl::aten::buildATen(bLoc);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqlen);
    at::Tensor atOut = impl::aten::buildATen(out);

    int block = 64;
    int batch = atBSeqLen.size(0);
    int head_num = logics.size(0);

    TritonKernelRunner_t kernel;
    const int gridX = batch;
    const int gridY = head_num;
    const int gridZ = 1;
    const int num_warps = 1;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 0;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    const char* kernel_name = "_fwd_kernel_destindex_copy_kv_0d1d2d3c4c5c6c7c8c9c";
    CUstream stream = static_cast<cudaStream_t>(stream_handle);
    void* logics_ptr = atLogics.data_ptr();
    void* v_ptr = atV.data_ptr();
    void* b_loc_ptr = atBLoc.data_ptr();
    void* b_start_loc_ptr = atBStartLoc.data_ptr();
    void* b_seq_len_ptr = atBSeqLen.data_ptr();
    void* out_ptr = atOut.data_ptr();

    unsigned int logics_stride[] = {atLogics.stride(0), atLogics.stride(1)};
    unsigned int v_stride[] = {atV.stride(0), atV.stride(1), atV.stride(2)};
    unsigned int b_loc_stride[] = {atBLoc.stride(0), atBLoc.stride(1)};
    unsigned int out_stride[] = {atOut.stride(0), atOut.stride(1), atOut.stride(2)};

    void** extra_param = NULL;
    void* kernel_params[] = {&logics_ptr,
                             &v_ptr,
                             &out_ptr,
                             &b_loc_ptr,
                             &b_start_loc_ptr,
                             &b_seq_len_ptr,
                             maxInputLen,
                             &logics_stride[0],
                             &logics_stride[1],
                             &v_stride[0],
                             &v_stride[1],
                             &v_stride[2],
                             &out_stride[0],
                             &out_stride[1],
                             &out_stride[2],
                             &b_loc_stride[0],
                             &b_loc_stride[1],
                             otherKVIndex};
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atQ = impl::aten::buildATen(q);
    at::Tensor atK = impl::aten::buildATen(k);
    at::Tensor atV = impl::aten::buildATen(v);
    at::Tensor atBStartLoc = impl::aten::buildATen(bStartLoc);
    at::Tensor atBSeqLen = impl::aten::buildATen(bSeqlen);
    at::Tensor atOut = impl::aten::buildATen(out);
    torch::Device q_device = atQ.device();
    at::Tensor atTmp = torch::empty({batch, head, max_input_len + 256}, q.options().dtype(torch::kFloat32));
    ;

    int block = 128;
    int dimK = atK.size(2);
    float smScale = 1.0 / std::sqrt(dimK);  // 计算scale系数
    int batch = atBSeqLen.size(0);
    int head_num = atQ.size(1);

    TritonKernelRunner_t kernel;
    const int gridX = batch;
    const int gridY = head_num;
    const int gridZ = maxInputLen / block;
    const int num_warps = dimK <= 64 ? 4 : 8;
    const int num_ctas = 1;
    const int clusterDimX = 1;
    const int clusterDimY = 1;
    const int clusterDimZ = 1;
    const int shared_memory = 0;

    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    const char* kernel_name = "_fwd_kernel_destindex_copy_kv_0d1d2d3c4c5c6c7c8c9c";
    CUstream stream = static_cast<cudaStream_t>(stream_handle);
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
    unsigned int tmp_stride[] = {atTmp.stride(0), atTmp.stride(1), atTmp.stride(2)};
    void** extra_param = NULL;
    void* kernel_params[] = {
        &q_ptr,       &k_ptr,         &v_ptr,         smScale,        &b_start_loc_ptr, &b_seq_len_ptr, &tmp_ptr,       &out_ptr,
        &q_stride[0], &q_stride[1],   &q_stride[2],   &k_stride[0],   &k_stride[1],     &k_stride[2],   &v_stride[0],   &v_stride[1],
        &v_stride[2], &out_stride[0], &out_stride[1], &out_stride[2], &tmp_stride[0],   &tmp_stride[1], &tmp_stride[2],
    };
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);
    impl::aten::sync(ctx);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

}  // extern "C"