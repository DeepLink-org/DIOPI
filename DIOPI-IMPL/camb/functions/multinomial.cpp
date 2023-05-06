/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

class CnnlRandGenerator final {
public:
    CnnlRandGenerator() { DIOPI_CHECKCNNL(cnnlRandCreateGenerator(&resource_, CNNL_RAND_RNG_FAST)); }
    ~CnnlRandGenerator() { DIOPI_CHECKCNNL(cnnlRandDestroyGenerator(resource_)); }
    cnnlRandGenerator_t& get() { return resource_; }

protected:
    cnnlRandGenerator_t resource_{0};
};

diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlRandGenerator cnnlGenerator;
    cnnlRandGenerator_t generator = cnnlGenerator.get();

    DiopiTensor input_tensor(input);
    DIOPI_CALL(autoCastTensorType(ctx, {&input_tensor}, {diopi_dtype_float16, diopi_dtype_float32}));
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor out_tensor(out);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size;
    DIOPI_CALLCNNL(cnnlGetRandGenerateMultinomialWorkspaceSize(handle, inputDesc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    if (out_tensor.dtype() == diopi_dtype_int32) {
        DIOPI_CALLCNNL(cnnlRandGenerateMultinomial_v2(
            handle, generator, inputDesc.get(), input_tensor.data(), replacement, false, nullptr, workspace, workspace_size, outDesc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp = requiresTensor(ctx, out_tensor.shape(), diopi_dtype_int32);
        CnnlTensorDesc out_tempDesc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlRandGenerateMultinomial_v2(handle,
                                                      generator,
                                                      inputDesc.get(),
                                                      input_tensor.data(),
                                                      replacement,
                                                      false,
                                                      nullptr,
                                                      workspace,
                                                      workspace_size,
                                                      out_tempDesc.get(),
                                                      out_temp.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
