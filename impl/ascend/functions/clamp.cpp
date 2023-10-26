/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <climits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
std::string getAclVersion() {
    int32_t majorVersion, minorVersion, patchVersion;
    CALL_ACLRT(aclrtGetVersion(&majorVersion, &minorVersion, &patchVersion));
    return std::to_string(majorVersion) + "." + std::to_string(minorVersion) + "." + std::to_string(patchVersion);
}
// diopiError_t broadcast(diopiContextHandle_t ctx, AscendTensor& out, const AscendTensor& input, const std::vector<int64_t>& size) {

//     auto ptr = const_cast<diopiConstTensorHandle_t>(out.tensorHandle());
//     out = AscendTensor(ptr);
//     return diopiSuccess;
// }

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    std::cout << getAclVersion() << std::endl;

    AscendTensor tem(input);
    std::cout << tem.numel() << std::endl;
    if (tem.numel() == 0) {
        return diopiSuccess;
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);

    diopiDtype_t outDtype, castType;
    diopiGetTensorDtype(out, &outDtype);
    if (isFloatingType(outDtype)) {
        castType = diopi_dtype_float32;
    } else {
        castType = diopi_dtype_int32;
    }

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, castType);

    if (min != nullptr) {
        diopiTensorHandle_t minTmp;
        makeTensorLike(ctx, &minTmp, input, castType);
        std::vector<int64_t> sizes;
        diopiSize_t diopiShape;
        diopiGetTensorShape(input, &diopiShape);
        std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
        sizes = std::move(shapeTmp);
        AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(min, castType).addConstInput(sizes).addOutput(minTmp).run();
        runner.addInput(minTmp, castType);
    } else {
        diopiTensorHandle_t minTmp;
        makeTensorLike(ctx, &minTmp, input, dtype);
        if (isIntegralType(dtype)) {
            fillTensor(ctx, minTmp, static_cast<float>(INT_MIN));
        } else {
            fillTensor(ctx, minTmp, static_cast<float>(-FLT_MAX));
        }
        runner.addInput(minTmp, castType);
    }
    if (max != nullptr) {
        diopiTensorHandle_t maxTmp;
        makeTensorLike(ctx, &maxTmp, input, castType);
        std::vector<int64_t> sizes;
        diopiSize_t diopiShape;
        diopiGetTensorShape(input, &diopiShape);
        std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
        sizes = std::move(shapeTmp);
        AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(max, castType).addConstInput(sizes).addOutput(maxTmp).run();
        runner.addInput(maxTmp, castType);
    } else {
        diopiTensorHandle_t maxTmp;
        makeTensorLike(ctx, &maxTmp, input, dtype);
        if (isIntegralType(dtype)) {
            fillTensor(ctx, maxTmp, static_cast<float>(INT_MAX));
        } else {
            fillTensor(ctx, maxTmp, static_cast<float>(FLT_MAX));
        }
        runner.addInput(maxTmp, castType);
    }
    runner.addOutput(out).run();

    return diopiSuccess;
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return diopiClamp(ctx, input, input, min, max);
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA----------------------------------------------------" << std::endl;
    AscendTensor tem(input);
    if (tem.numel() > 1) {
        diopiTensorHandle_t minTmp = nullptr, maxTmp = nullptr;
        if (min != nullptr) makeTensorFromScalar(ctx, min, &minTmp, diopi_dtype_float64, diopi_device);
        if (max != nullptr) makeTensorFromScalar(ctx, max, &maxTmp, diopi_dtype_float64, diopi_device);
        diopiClamp(ctx, out, input, minTmp, maxTmp);
        return diopiSuccess;
    }

    diopiDtype_t outDtype;
    diopiGetTensorDtype(out, &outDtype);

    diopiDtype_t dtype;
    if (isFloatingType(outDtype)) {
        dtype = diopi_dtype_float32;
    } else {
        dtype = diopi_dtype_int32;
    }
    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, dtype);
    if (min != nullptr) {
        runner.addConstInput(*min, dtype);
    } else {
        if (isIntegralType(dtype)) {
            runner.addConstInput(INT_MIN, dtype);
        } else {
            runner.addConstInput(-FLT_MAX, dtype);
        }
    }
    if (max != nullptr) {
        runner.addConstInput(*max, dtype);
    } else {
        if (isIntegralType(dtype)) {
            runner.addConstInput(INT_MAX, dtype);
        } else {
            runner.addConstInput(FLT_MAX, dtype);
        }
    }
    runner.addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    return diopiClampScalar(ctx, input, input, min, max);
}

}  // namespace ascend
}  // namespace impl
