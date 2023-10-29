/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <climits>
#include <limits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    dtype = diopi_dtype_float64;

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, dtype);

    diopiTensorHandle_t minTmp, maxTmp, boolOut;
    makeTensorLike(ctx, &minTmp, input, dtype);
    makeTensorLike(ctx, &maxTmp, input, dtype);
    makeTensorLike(ctx, &boolOut, input, diopi_dtype_bool);

    std::vector<int64_t> sizes;
    diopiSize_t diopiShape;
    diopiGetTensorShape(input, &diopiShape);
    std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
    sizes = std::move(shapeTmp);

    if (min != nullptr) {
        if (!sizes.empty() > 0) {
            AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(min, dtype).addConstInput(sizes).addOutput(minTmp).run();
        } else {
            minTmp = const_cast<diopiTensorHandle_t>(min);
        }
    } else {
        fillTensor(ctx, minTmp, -std::numeric_limits<double>::max());
    }

    if (max != nullptr) {
        if (!sizes.empty() > 0) {
            AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(max, dtype).addConstInput(sizes).addOutput(maxTmp).run();
        } else {
            maxTmp = const_cast<diopiTensorHandle_t>(max);
        }
    } else {
        fillTensor(ctx, maxTmp, std::numeric_limits<double>::max());
    }

    diopiLt(ctx, boolOut, maxTmp, minTmp);
    diopiMaskedFill(ctx, minTmp, minTmp, boolOut, maxTmp);

    runner.addInput(minTmp, dtype).addInput(maxTmp, dtype).addOutput(out).run();

    return diopiSuccess;
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return diopiClamp(ctx, input, input, min, max);
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    diopiDtype_t dtype;
    dtype = diopi_dtype_float64;

    AclOpRunner<3, 1> runner("ClipByValue", ctx);
    runner.addInput(input, dtype);

    if (min != nullptr && max != nullptr) {
        double minn = getValue<double>(min);
        double maxn = getValue<double>(max);
        if (maxn < minn) {
            runner.addConstInput(*max, dtype).addConstInput(*max, dtype).addOutput(out).run();
            return diopiSuccess;
        }
    }
    if (min != nullptr) {
        runner.addConstInput(*min, dtype);
    } else {
        runner.addConstInput(-std::numeric_limits<double>::max(), dtype);
    }

    if (max != nullptr) {
        runner.addConstInput(*max, dtype);
    } else {
        runner.addConstInput(std::numeric_limits<double>::max(), dtype);
    }

    runner.addOutput(out).run();
    return diopiSuccess;
}
diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    return diopiClampScalar(ctx, input, input, min, max);
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    return diopiClampMinScalar(ctx, input, input, min);
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    return diopiClampMin(ctx, input, input, min);
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    return diopiClampScalar(ctx, out, input, min, nullptr);
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    return diopiClamp(ctx, out, input, min, nullptr);
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    return diopiClampMaxScalar(ctx, input, input, max);
}

DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    return diopiClampMax(ctx, input, input, max);
}

DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    return diopiClampScalar(ctx, out, input, nullptr, max);
}

DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    return diopiClamp(ctx, out, input, nullptr, max);
}

}  // namespace ascend
}  // namespace impl
