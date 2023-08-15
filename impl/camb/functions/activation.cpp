/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>
#include <memory>

#include "../common/common.hpp"
namespace impl {
namespace camb {

namespace {
class CnnlAttribute {
public:
    template <typename T>
    void set(std::string key, T value) {
        auto pHolder = std::make_shared<ValueHolder<T>>(value);
        mData_[key] = pHolder;
    }

    template <typename T>
    T get(std::string key, T defaultValue) const {
        auto iter = mData_.find(key);
        if (iter != mData_.end()) {
            const ValueHolderBase* holder = iter->second.get();
            if (holder->getTypeInfo() == typeid(T)) {
                const ValueHolder<T>* typedHolder = static_cast<const ValueHolder<T>*>(holder);
                return typedHolder->get();
            }
        }
        return defaultValue;
    }

private:
    class ValueHolderBase {
    public:
        virtual ~ValueHolderBase() = default;
        virtual const std::type_info& getTypeInfo() const = 0;
    };

    template <typename T>
    class ValueHolder : public ValueHolderBase {
    public:
        explicit ValueHolder(T value) : mValue_(value) {}
        const std::type_info& getTypeInfo() const override { return typeid(T); }
        T get() const { return mValue_; }

    private:
        T mValue_;
    };

    std::unordered_map<std::string, std::shared_ptr<ValueHolderBase>> mData_;
};

diopiError_t cnnlActivationInternal(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor out, CnnlAttribute attr) {
    auto handle = cnnlHandlePool.get(ctx);
    auto mode = attr.get<cnnlActivationMode_t>("mode", CNNL_ACTIVATION_SIGMOID);
    auto perf = attr.get<cnnlActivationPreference_t>("perf", CNNL_ACTIVATION_HIGH_PRECISION);
    auto nanProp = attr.get<cnnlNanPropagation_t>("nan", CNNL_NOT_PROPAGATE_NAN);

    float coef = attr.get("coef", 0.0f);
    int slicedDim = attr.get("sliced_dim", 0);
    float gamma = attr.get("gamma", 0.0f);
    float scale = attr.get("scale", 0.0f);
    bool isResult = attr.get("is_result", false);
    bool approximate = attr.get("approximate", false);
    void* alpha = attr.get("alpha", nullptr);
    void* beta = attr.get("beta", nullptr);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activationDesc;
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v6(activationDesc.get(), mode, perf, nanProp, coef, slicedDim, gamma, scale, isResult, approximate));

    std::vector<DiopiTensor*> inputs{&input};
    DIOPI_CALL(autoCastTensorType(ctx, inputs, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor tempOutput = out;
    DIOPI_CALL(dataTypeCast(ctx, tempOutput, input.dtype()));

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(tempOutput, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlActivationForward(handle, activationDesc.get(), alpha, inputDesc.get(), input.data(), beta, outputDesc.get(), tempOutput.data()));
    DIOPI_CALL(dataTypeCast(ctx, out, tempOutput));

    return diopiSuccess;
}

diopiError_t cnnlActivationBackwardInternal(diopiContextHandle_t ctx, DiopiTensor gradInput, DiopiTensor gradOutput, DiopiTensor input, DiopiTensor output,
                                            CnnlAttribute attr) {
    // if (!input.defined()) {
    //     return diopiSuccess;
    // }
    auto handle = cnnlHandlePool.get(ctx);
    auto mode = attr.get<cnnlActivationMode_t>("mode", CNNL_ACTIVATION_SIGMOID);
    auto perf = attr.get<cnnlActivationPreference_t>("perf", CNNL_ACTIVATION_HIGH_PRECISION);
    auto nanProp = attr.get<cnnlNanPropagation_t>("perf", CNNL_NOT_PROPAGATE_NAN);  // relu relu6

    float coef = attr.get("coef", 0.0f);
    int slicedDim = attr.get("sliced_dim", 0);
    float gamma = attr.get("gamma", 0.0f);
    float scale = attr.get("scale", 0.0f);
    bool isResult = attr.get("is_result", true);
    bool approximate = attr.get("approximate", false);
    void* alpha = attr.get("alpha", nullptr);
    void* beta = attr.get("beta", nullptr);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activationDesc;
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v6(activationDesc.get(), mode, perf, nanProp, coef, slicedDim, gamma, scale, isResult, approximate));
    std::vector<DiopiTensor*> inputs{&gradOutput};
    if (input.defined()) {
        inputs.push_back(&input);
    }
    if (output.defined()) {
        inputs.push_back(&output);
    }

    std::set<diopiDtype_t> supportDtype{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, inputs, supportDtype));
    DiopiTensor tempGradInput = gradInput;
    DIOPI_CALL(dataTypeCast(ctx, tempGradInput, gradOutput.dtype()));

    CnnlTensorDesc gradInputDesc(tempGradInput, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutput, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc inputDesc, outputDesc;
    if (input.defined()) {
        DIOPI_CALL(inputDesc.set(input, CNNL_LAYOUT_ARRAY));
    }
    if (output.defined()) {
        DIOPI_CALL(outputDesc.set(output, CNNL_LAYOUT_ARRAY));
    }

    DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                          activationDesc.get(),
                                          alpha,
                                          outputDesc.get(),
                                          output.defined() ? output.data() : nullptr,
                                          gradOutputDesc.get(),
                                          gradOutput.data(),
                                          inputDesc.get(),
                                          input.defined() ? input.data() : nullptr,
                                          beta,
                                          gradInputDesc.get(),
                                          tempGradInput.data()));
    DIOPI_CALL(dataTypeCast(ctx, gradInput, tempGradInput));
    return diopiSuccess;
}

}  // namespace

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_RELU);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_RELU);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, inputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, inputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                             diopiConstTensorHandle_t output) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor outputTensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnlActivationBackwardInternal(ctx, gradInputTensor, gradOutputTensor, {}, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SILU);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SILU);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, inputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                          diopiConstTensorHandle_t input) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SILU);
    DIOPI_CALL(cnnlActivationBackwardInternal(ctx, gradInputTensor, gradOutputTensor, inputTensor, {}, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, inputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                          diopiConstTensorHandle_t output) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor outputTensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnlActivationBackwardInternal(ctx, gradInputTensor, gradOutputTensor, {}, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_GELU);
    if (strcmp(approximate, "tanh") == 0) {
        attr.set("approximate", true);
    }
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));

    return diopiSuccess;
}

extern "C" diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                          diopiConstTensorHandle_t input, const char* approximate) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_GELU);
    if (strcmp(approximate, "tanh") == 0) {
        attr.set("approximate", true);
    }

    DIOPI_CALL(cnnlActivationBackwardInternal(ctx, gradInputTensor, gradOutputTensor, inputTensor, {}, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    float coefVal = DiopiDataType::isInteger(negativeSlope->stype) ? negativeSlope->ival : negativeSlope->fval;
    attr.set("coef", coefVal);
    attr.set("mode", CNNL_ACTIVATION_LEAKYRELU);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negativeSlope) {
    DiopiTensor inputTensor(input);
    CnnlAttribute attr;
    float coefVal = DiopiDataType::isInteger(negativeSlope->stype) ? negativeSlope->ival : negativeSlope->fval;
    attr.set("coef", coefVal);
    attr.set("mode", CNNL_ACTIVATION_LEAKYRELU);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, inputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                               diopiConstTensorHandle_t input, const diopiScalar_t* negativeSlope, bool inputIsResult) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    float coefVal = DiopiDataType::isInteger(negativeSlope->stype) ? negativeSlope->ival : negativeSlope->fval;
    attr.set("coef", coefVal);
    attr.set("mode", CNNL_ACTIVATION_LEAKYRELU);
    DIOPI_CALL(cnnlActivationBackwardInternal(ctx, gradInputTensor, gradOutputTensor, inputTensor, {}, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_HARDSWISH);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, outputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_HARDSWISH);
    DIOPI_CALL(cnnlActivationInternal(ctx, inputTensor, inputTensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                               diopiConstTensorHandle_t input) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_HARDSWISH);
    DIOPI_CALL(cnnlActivationBackwardInternal(ctx, gradInputTensor, gradOutputTensor, inputTensor, {}, attr));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
