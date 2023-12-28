/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/diopirt.h>

#include "../common/acloprunner.hpp"
#include "../common/format_helper.h"

namespace impl {
namespace ascend {
void formatCastInsideGroup(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t in) {
    AscendTensor tensorIn(in);
    AscendTensor tensorOut(out);
    AclOpRunner<1, 1>("Identity", ctx).addInput(tensorIn).addOutput(tensorOut).run();
}
void formatCastBetweenGroup(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t in) {
    diopiStorageDesc_t descIn;
    diopiGetTensorStorageDesc(in, &descIn);
    diopiStorageDesc_t descOut;
    diopiGetTensorStorageDesc(out, &descOut);
    bool isInputBaseFormat = FormatHelper::isBaseFormat(descIn.format);
    bool isOutputBaseFormat = FormatHelper::isBaseFormat(descOut.format);
    if (isInputBaseFormat && !isOutputBaseFormat) {
        diopiMemoryFormat_t inputFormatTmp = descIn.format;
        descIn.format = FormatHelper::getDiopiBaseFormat(descOut.format);
        diopiSetTensorStorageDesc(in, descIn);
        formatCastInsideGroup(ctx, out, in);
        descIn.format = inputFormatTmp;
        diopiSetTensorStorageDesc(in, descIn);
    } else if (!isInputBaseFormat && isOutputBaseFormat) {
        diopiMemoryFormat_t outFormatTmp = descOut.format;
        descOut.format = FormatHelper::getDiopiBaseFormat(descIn.format);
        diopiSetTensorStorageDesc(out, descOut);
        formatCastInsideGroup(ctx, out, in);
        descOut.format = outFormatTmp;
        diopiSetTensorStorageDesc(out, descOut);
    } else {
        ASCEND_CHECK_ABORT(false, "format cast not support");
    }
}

diopiError_t diopiFormatCast(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t in, diopiMemoryFormat_t targetFormat) {
    AscendTensor tensorIn(in);
    if (tensorIn.storageFormat() == targetFormat) {
        *out = in;
        return diopiSuccess;
    }
    if (FormatHelper::isBaseFormat(tensorIn.storageFormat()) && FormatHelper::isBaseFormat(targetFormat)) {
        diopiStorageDesc_t desc;
        diopiGetTensorStorageDesc(in, &desc);
        desc.format = targetFormat;
        *out = in;
        diopiSetTensorStorageDesc(*out, desc);
        return diopiSuccess;
    }
    std::vector<int64_t> storageSizesOut = FormatHelper::getStorageSizes(targetFormat, tensorIn.shape());
    diopiStorageDesc_t descOut;
    descOut.sizes.data = storageSizesOut.data();
    descOut.sizes.len = storageSizesOut.size();
    descOut.format = targetFormat;
    diopiRequireTensor(ctx, out, &descOut.sizes, nullptr, tensorIn.dtype(), tensorIn.device());
    diopiSetTensorStorageDesc(*out, descOut);
    // set tensor metadata
    diopiCopyTensorMetaData(*out, in);
    if (FormatHelper::getDiopiBaseFormat(targetFormat) != FormatHelper::getDiopiBaseFormat(tensorIn.storageFormat())) {
        formatCastBetweenGroup(ctx, *out, in);
    } else {
        formatCastInsideGroup(ctx, *out, in);
    }
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
