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
    CnnlRandGenerator() { DIOPI_CHECKCNNL(cnnlRandCreateGenerator(&resource_, CNNL_RAND_RNG_MTGP32)); }
    ~CnnlRandGenerator() { DIOPI_CHECKCNNL(cnnlRandDestroyGenerator(resource_)); }
    cnnlRandGenerator_t& get() { return resource_; }

private:
    cnnlRandGenerator_t resource_{nullptr};
};

diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numSamples, bool replacement,
                              diopiGeneratorHandle_t gen) {
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
    DIOPI_CALL_CNNL(cnnlGetRandGenerateMultinomialWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    diopiTensorHandle_t stateHandle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, gen, &stateHandle));
    void* statePtr = nullptr;
    DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));
    DIOPI_CALL_CNNL(cnnlRandGenerateMultinomial_v2(
        handle, generator, inputDesc.get(), inputTensor.data(), replacement, false, statePtr, workspace, workspaceSize, outDesc.get(), outTemp.data()));
    DIOPI_CALL(diopiGeneratorSetState(gen, stateHandle));
    if (outTensor.dtype() != outTemp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTemp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
