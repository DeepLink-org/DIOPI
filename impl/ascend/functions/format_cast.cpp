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
    AscendTensor tensor_in(in);
    AscendTensor tensor_out(out);
    AclOpRunner<1, 1>("Identity", ctx).addInput(tensor_in).addOutput(tensor_out).run();
}
void formatCastBetweenGroup(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t in) {
    diopiStorageDesc_t desc_in;
    diopiGetTensorStorageDesc(in, &desc_in);
    diopiStorageDesc_t desc_out;
    diopiGetTensorStorageDesc(out, &desc_out);
    bool isInputBaseFormat = FormatHelper::isBaseFormat(desc_in.format);
    bool isOutputBaseFormat = FormatHelper::isBaseFormat(desc_out.format);
    if (isInputBaseFormat && !isOutputBaseFormat) {
        diopiMemoryFormat_t input_format_tmp = desc_in.format;
        desc_in.format = FormatHelper::getDiopiBaseFormat(desc_out.format);
        diopiSetTensorStorageDesc(in, desc_in);
        formatCastInsideGroup(ctx, out, in);
        desc_in.format = input_format_tmp;
        diopiSetTensorStorageDesc(in, desc_in);
    } else if (!isInputBaseFormat && isOutputBaseFormat) {
        diopiMemoryFormat_t out_format_tmp = desc_out.format;
        desc_out.format = FormatHelper::getDiopiBaseFormat(desc_in.format);
        diopiSetTensorStorageDesc(out, desc_out);
        formatCastInsideGroup(ctx, out, in);
        desc_out.format = out_format_tmp;
        diopiSetTensorStorageDesc(out, desc_out);
    } else {
        ASCEND_CHECK_ABORT(false, "format cast not support");
    }
}

diopiError_t diopiFormatCast(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t in, diopiMemoryFormat_t target_format) {
    AscendTensor tensor_in(in);
    if (tensor_in.storageFormat() == target_format) {
        *out = in;
        return diopiSuccess;
    }
    if (FormatHelper::isBaseFormat(tensor_in.storageFormat()) && FormatHelper::isBaseFormat(target_format)) {
        diopiStorageDesc_t desc;
        diopiGetTensorStorageDesc(in, &desc);
        desc.format = target_format;
        *out = in;
        diopiSetTensorStorageDesc(*out, desc);
        return diopiSuccess;
    }
    std::vector<int64_t> storage_sizes_out = FormatHelper::getStorageSizes(target_format, tensor_in.shape());
    diopiStorageDesc_t desc_out;
    desc_out.sizes.data = storage_sizes_out.data();
    desc_out.sizes.len = storage_sizes_out.size();
    desc_out.format = target_format;
    diopiRequireTensor(ctx, out, &desc_out.sizes, nullptr, tensor_in.dtype(), tensor_in.device());
    diopiSetTensorStorageDesc(*out, desc_out);
    // set tensor metadata
    diopiCopyTensorMetaData(*out, in);
    if (FormatHelper::getDiopiBaseFormat(target_format) != FormatHelper::getDiopiBaseFormat(tensor_in.storageFormat())) {
        formatCastBetweenGroup(ctx, *out, in);
    } else {
        formatCastInsideGroup(ctx, *out, in);
    }
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
