/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <diopi/functions_lightllm.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include "cuda.h"
#include "context.h"
#include "helper.hpp"

#define checkCudaErrors(x) \
  { \
  CUresult err = x;       \
  if (err != CUDA_SUCCESS) { \
    std::cout << "checkCudaErrors:"<< __FILE__ << ":" << __LINE__ << #x << ": " << err << std::endl; \
  }             \
  }


namespace {

    bool read_file(std::string filename, std::vector<char>& buffer)
{
    std::ifstream infile(filename.c_str(), std::ifstream::binary);
    if (!infile.is_open())
    {
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
            std::vector<char> buffer;
            CuKernelModuleLoader() {
                std::string fatbin_path = std::getenv("HOME") + std::string("/.triton/diopi_triton_kernels.fatbin");
                read_file(fatbin_path, buffer);
                checkCudaErrors(cuModuleLoadFatBinary(&cudaModule_, buffer.data()));
            }

            ~CuKernelModuleLoader() {
                checkCudaErrors(cuModuleUnload(cudaModule_));
            }
        public:
        static CUmodule& get() {
            static CuKernelModuleLoader loader;
            return loader.cudaModule_;
        }
    };

    template<int kernel_id>
    class TritonKernelRunner {
        public:

        void run(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, const char* kernel_name, CUstream stream, void** kernelParams, void** extra) {
            CUfunction  function;
            checkCudaErrors(cuModuleGetFunction(&function, CuKernelModuleLoader::get(), kernel_name));
            if (gridX * gridY * gridZ > 0) {
                if (num_ctas == 1) {
                    checkCudaErrors(cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, kernelParams, extra));
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

                    checkCudaErrors(cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, shared_memory, stream, kernelParams, extra));
                }
            }
        }
    };

    #define TritonKernelRunner_t TritonKernelRunner<__LINE__>

    unsigned int next_power_of_2(unsigned int x) {
        return std::pow(2, std::ceil(std::log(x) / std::log(2)));
    }

} // namespace

extern "C" {

diopiError_t diopiDestindexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
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

    const char* kernel_name = "_fwd_kernel_destindex_copy_kv_0d1d2d3de4de5c6de7de8c9de";
    CUstream stream = NULL;
    void* k_ptr = atK.data_ptr();
    void* out_ptr = atOut.data_ptr();
    void* destLoc_ptr = atDestLoc.data_ptr();
    unsigned int k_stride[] = {atK.stride(0), atK.stride(1), atK.stride(2)};
    unsigned int out_stride[] = {atOut.stride(0), atOut.stride(1), atOut.stride(2)};
    void** extra_param = NULL;
    void *kernel_params[] = { &k_ptr, &destLoc_ptr, &out_ptr , &head_dim, &k_stride[0], &k_stride[1], &k_stride[2], &out_stride[0], &out_stride[1], &out_stride[2]};
    kernel.run(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, kernel_name, stream, kernel_params, extra_param);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqlen,
                                          int maxInputLen) {
    return diopiSuccess;
}

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqlen,
                                               int maxInputLen, int otherKVIndex) {
    return diopiSuccess;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqlen, int maxInputLen) {
    return diopiSuccess;
}

}  // extern "C"
