/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

class CnnlRandGenerator final {
public:
    CnnlRandGenerator() { DIOPI_CHECKCNNL(cnnlRandCreateGenerator(&resource_, CNNL_RAND_RNG_FAST)); }
    ~CnnlRandGenerator() { DIOPI_CHECKCNNL(cnnlRandDestroyGenerator(resource_)); }
    cnnlRandGenerator_t& get() { return resource_; }

private:
    cnnlRandGenerator_t resource_{nullptr};
};

diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples, bool replacement) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlRandGenerator cnnlGenerator;
    cnnlRandGenerator_t generator = cnnlGenerator.get();

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor outTemp = outTensor;
    DIOPI_CALL(autoCastTensorType(ctx, {&inputTensor}, {diopi_dtype_float16, diopi_dtype_float32}));
    DIOPI_CALL(autoCastTensorType(ctx, {&outTemp}, {diopi_dtype_int32}));
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize;
    DIOPI_CALLCNNL(cnnlGetRandGenerateMultinomialWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlRandGenerateMultinomial_v2(
        handle, generator, inputDesc.get(), inputTensor.data(), replacement, false, nullptr, workspace, workspaceSize, outDesc.get(), outTemp.data()));
    if (outTensor.dtype() != outTemp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTemp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
