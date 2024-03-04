/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace {
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& normalOutNpuNocheck(at::Tensor& result, c10::optional<at::Generator> gen) {
    auto genDefault = gen.value().get<at_npu::NPUGeneratorImpl>();

    auto pair = genDefault->philox_engine_inputs(10);
    const int64_t seed = pair.first;
    const int64_t offset = pair.second;

    at::SmallVector<int64_t, N> key = {seed};
    at::SmallVector<int64_t, N> counter = {0, offset};
    const int32_t alg = 1;

    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessRandomNormalV2")
        .Input(result.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(key, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string) "uint64")
        .Input(counter, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string) "uint64")
        .Input(at::Scalar(alg), at::ScalarType::Int)
        .Output(result)
        .Attr("dtype", result.scalar_type())
        .Run();
    return result;
}
}  // namespace

at::Tensor& normal_(at::Tensor& self, double mean = 0, double std = 1, c10::optional<at::Generator> generator = c10::nullopt);
at::Tensor& normal_out(const at::Tensor& mean, const at::Tensor& std, c10::optional<at::Generator> generator, at::Tensor& out);
at::Tensor& normal_out(const at::Tensor& mean, double std, c10::optional<at::Generator> generator, at::Tensor& out);
at::Tensor& normal_out(double mean, const at::Tensor& std, c10::optional<at::Generator> generator, at::Tensor& out);
at::Tensor& normal_out(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor& out);

namespace OP_IMPL_NS {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator);
    if (out == nullptr || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::normal_out(mean, std, outAt.sizes(), generatorAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(inout, generator);
    if (inout == nullptr || inoutAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::normal_(inoutAt, mean, std, generatorAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std,
                                     diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator, std);
    if (out == nullptr || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::normal_out(mean, stdAt, generatorAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std,
                               diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator, mean, std);
    if (out == nullptr || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::normal_out(meanAt, stdAt, generatorAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std,
                                     diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator, mean);
    if (out == nullptr || outAt.numel() <= 0) {
        return diopiSuccess;
    }
    op_api::normal_out(meanAt, std, generatorAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
