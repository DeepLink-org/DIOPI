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

private:
    cnnlRandGenerator_t resource_{0};
};

diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlRandGenerator cnnlGenerator;
    cnnlRandGenerator_t generator = cnnlGenerator.get();

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    DiopiTensor out_temp = out_tensor;
    DIOPI_CALL(autoCastTensorType(ctx, {&input_tensor, &out_temp}, {diopi_dtype_float32}));
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_temp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size;
    DIOPI_CALLCNNL(cnnlGetRandGenerateMultinomialWorkspaceSize(handle, inputDesc.get(), &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlRandGenerateMultinomial_v2(
        handle, generator, inputDesc.get(), input_tensor.data(), replacement, false, nullptr, workspace, workspace_size, outDesc.get(), out_temp.data()));
    if (out_tensor.dtype() != out_temp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
