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

    float coef = attr.get("coef", 0.0);
    int sliced_dim = attr.get("sliced_dim", 0);
    float gamma = attr.get("gamma", 0.0);
    float scale = attr.get("scale", 0.0);
    void* alpha = attr.get("alpha", nullptr);
    void* beta = attr.get("beta", nullptr);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activation_desc;
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v4(activation_desc.get(), mode, perf, nan_prop, coef, sliced_dim, gamma, scale));

    std::vector<DiopiTensor*> inputs{&input};
    autoCastTensorType(ctx, inputs, {diopi_dtype_float16, diopi_dtype_float32});
    DiopiTensor temp_input = *inputs[0];
    DiopiTensor temp_output = dataTypeCast(ctx, out, temp_input.dtype());

    CnnlTensorDesc input_desc(temp_input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(temp_output, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(
        cnnlActivationForward(handle, activation_desc.get(), alpha, input_desc.get(), temp_input.data(), beta, output_desc.get(), temp_output.data()));
    dataTypeCast(ctx, out, temp_output);

    return diopiSuccess;
}

diopiError_t cnnl_activation_backward_internal(diopiContextHandle_t ctx, DiopiTensor grad_input, DiopiTensor grad_output, DiopiTensor output,
                                               CnnlAttribute attr) {
    auto handle = cnnlHandlePool.get(ctx);
    auto mode = attr.get<cnnlActivationMode_t>("mode", CNNL_ACTIVATION_SIGMOID);
    auto perf = attr.get<cnnlActivationPreference_t>("perf", CNNL_ACTIVATION_HIGH_PRECISION);
    auto nan_prop = attr.get<cnnlNanPropagation_t>("perf", CNNL_NOT_PROPAGATE_NAN);  // relu relu6

    float coef = attr.get("coef", 0.0);
    int sliced_dim = attr.get("sliced_dim", 0);
    float gamma = attr.get("gamma", 0.0);
    float scale = attr.get("scale", 0.0);
    bool is_result = attr.get("is_result", true);
    bool approximate = attr.get("approximate", true);
    void* alpha = attr.get("alpha", nullptr);
    void* beta = attr.get("beta", nullptr);

    CnnlResourceGuard<cnnlActivationDescriptor_t, cnnlCreateActivationDescriptor, cnnlDestroyActivationDescriptor> activation_desc;
    DIOPI_CALLCNNL(cnnlSetActivationDescriptor_v6(activation_desc.get(), mode, perf, nan_prop, coef, sliced_dim, gamma, scale, is_result, approximate));

    std::vector<DiopiTensor*> inputs{&output, &grad_output};
    autoCastTensorType(ctx, inputs, {diopi_dtype_float16, diopi_dtype_float32});
    DiopiTensor temp_output = *inputs[0];
    DiopiTensor temp_grad_output = *inputs[1];
    DiopiTensor temp_grad_input = dataTypeCast(ctx, grad_input, temp_output.dtype());

    CnnlTensorDesc grad_input_desc(temp_grad_input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_output_desc(temp_grad_output, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(temp_output, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlActivationBackward(handle,
                                          activation_desc.get(),
                                          alpha,
                                          output_desc.get(),
                                          output.data(),
                                          grad_output_desc.get(),
                                          grad_output.data(),
                                          grad_input_desc.get(),
                                          grad_input.data(),
                                          beta,
                                          grad_input_desc.get(),
                                          grad_input.data()));
    dataTypeCast(ctx, grad_input, temp_grad_input);
    return diopiSuccess;
}

}  // namespace

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_RELU);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_RELU);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_output_tensor = DiopiTensor(grad_output);
    auto output_tensor = DiopiTensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_SIGMOID);
    cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, output_tensor, attr);
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, output_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    DIOPI_CALL(cnnl_activation_internal(ctx, input_tensor, input_tensor, attr));
    return diopiSuccess;
}

extern "C" diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto grad_input_tensor = DiopiTensor(grad_input);
    auto grad_output_tensor = DiopiTensor(grad_output);
    auto output_tensor = DiopiTensor(output);

    CnnlAttribute attr;
    attr.set("mode", CNNL_ACTIVATION_TANH);
    cnnl_activation_backward_internal(ctx, grad_input_tensor, grad_output_tensor, output_tensor, attr);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
