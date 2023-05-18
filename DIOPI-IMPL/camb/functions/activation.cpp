/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

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
        m_data[key] = pHolder;
    }

    template <typename T>
    T get(std::string key, T defaultValue) const {
        auto iter = m_data.find(key);
        if (iter != m_data.end()) {
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
        virtual ~ValueHolderBase() {}
        virtual const std::type_info& getTypeInfo() const = 0;
    };

    template <typename T>
    class ValueHolder : public ValueHolderBase {
    public:
        explicit ValueHolder(T value) : m_value(value) {}
        const std::type_info& getTypeInfo() const override { return typeid(T); }
        T get() const { return m_value; }

    private:
        T m_value;
    };

    std::unordered_map<std::string, std::shared_ptr<ValueHolderBase>> m_data;
};

diopiError_t cnnl_activation_internal(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor out, CnnlAttribute attr) {
    auto handle = cnnlHandlePool.get(ctx);
    auto mode = attr.get<cnnlActivationMode_t>("mode", CNNL_ACTIVATION_SIGMOID);
    auto perf = attr.get<cnnlActivationPreference_t>("perf", CNNL_ACTIVATION_HIGH_PRECISION);
    auto nan_prop = attr.get<cnnlNanPropagation_t>("nan", CNNL_NOT_PROPAGATE_NAN);

    float coef = attr.get("coef", 0.0f);
    int sliced_dim = attr.get("sliced_dim", 0);
    float gamma = attr.get("gamma", 0.0f);
    float scale = attr.get("scale", 0.0f);
    bool is_result = attr.get("is_result", false);
    bool approximate = attr.get("approximate", false);
    void* alpha = attr.get("alpha", nullptr);
    void* beta = attr.get("beta", nullptr);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activation_desc;
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v6(activation_desc.get(), mode, perf, nan_prop, coef, sliced_dim, gamma, scale, is_result, approximate));

    std::vector<DiopiTensor*> inputs{&input};
    DIOPI_CALL(autoCastTensorType(ctx, inputs, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor temp_output = out;
    DIOPI_CALL(dataTypeCast(ctx, temp_output, input.dtype()));

    CnnlTensorDesc input_desc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(temp_output, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlActivationForward(handle, activation_desc.get(), alpha, input_desc.get(), input.data(), beta, output_desc.get(), temp_output.data()));
    DIOPI_CALL(dataTypeCast(ctx, out, temp_output));

    return diopiSuccess;
}

diopiError_t cnnl_activation_backward_internal(diopiContextHandle_t ctx, DiopiTensor grad_input, DiopiTensor grad_output, DiopiTensor input, DiopiTensor output,
                                               CnnlAttribute attr) {
    auto handle = cnnlHandlePool.get(ctx);
    auto mode = attr.get<cnnlActivationMode_t>("mode", CNNL_ACTIVATION_SIGMOID);
    auto perf = attr.get<cnnlActivationPreference_t>("perf", CNNL_ACTIVATION_HIGH_PRECISION);
    auto nan_prop = attr.get<cnnlNanPropagation_t>("perf", CNNL_NOT_PROPAGATE_NAN);  // relu relu6

    float coef = attr.get("coef", 0.0f);
    int sliced_dim = attr.get("sliced_dim", 0);
    float gamma = attr.get("gamma", 0.0f);
    float scale = attr.get("scale", 0.0f);
    bool is_result = attr.get("is_result", true);
    bool approximate = attr.get("approximate", false);
    void* alpha = attr.get("alpha", nullptr);
    void* beta = attr.get("beta", nullptr);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activation_desc;
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v6(activation_desc.get(), mode, perf, nan_prop, coef, sliced_dim, gamma, scale, is_result, approximate));
    std::vector<DiopiTensor*> inputs{&grad_output};
    if (input.defined()) {
        inputs.push_back(&input);
    }
    if (output.defined()) {
        inputs.push_back(&output);
    }

    std::set<diopiDtype_t> support_dtype{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, inputs, support_dtype));
    DiopiTensor temp_grad_input = grad_input;
    DIOPI_CALL(dataTypeCast(ctx, temp_grad_input, grad_output.dtype()));

    CnnlTensorDesc grad_input_desc(temp_grad_input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_output_desc(grad_output, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc input_desc, output_desc;
    if (input.defined()) {
        DIOPI_CALL(input_desc.set(input, CNNL_LAYOUT_ARRAY));
    }
    if (output.defined()) {
        DIOPI_CALL(output_desc.set(output, CNNL_LAYOUT_ARRAY));
    }

    DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                          activation_desc.get(),
                                          alpha,
                                          output_desc.get(),
                                          output.defined() ? output.data() : nullptr,
                                          grad_output_desc.get(),
                                          grad_output.data(),
                                          input_desc.get(),
                                          input.defined() ? input.data() : nullptr,
                                          beta,
                                          grad_input_desc.get(),
                                          temp_grad_input.data()));
    DIOPI_CALL(dataTypeCast(ctx, grad_input, temp_grad_input));
    return diopiSuccess;
}

}  // namespace

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_RELU);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_RELU);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor output_tensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, {}, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor output_tensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, {}, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_GELU);
    if (approximate == "tanh") {
        attr.set("approximate", true);
    }
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));

    return diopiSuccess;
}

extern "C" diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t input, const char* approximate) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor input_tensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_GELU);
    if (approximate == "tanh") {
        attr.set("approximate", true);
    }

    DIOPI_CALL(cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, input_tensor, {}, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);

    CnnlAttribute attr;
    float coef_val = DiopiDataType::isInteger(negative_slope->stype) ? negative_slope->ival : negative_slope->fval;
    attr.set("coef", coef_val);
    attr.set("mode", CNNL_ACTIVATION_LEAKYRELU);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    CnnlAttribute attr;
    float coef_val = DiopiDataType::isInteger(negative_slope->stype) ? negative_slope->ival : negative_slope->fval;
    attr.set("coef", coef_val);
    attr.set("mode", CNNL_ACTIVATION_LEAKYRELU);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor input_tensor(input);

    CnnlAttribute attr;
    float coef_val = DiopiDataType::isInteger(negative_slope->stype) ? negative_slope->ival : negative_slope->fval;
    attr.set("coef", coef_val);
    attr.set("mode", CNNL_ACTIVATION_LEAKYRELU);
    DIOPI_CALL(cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, input_tensor, {}, attr));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
