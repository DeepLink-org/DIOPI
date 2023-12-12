/**************************************************************************************************
 * Copyright 2022 Enflame. All Rights Reserved.
 * License: BSD 3-Clause
 *
 *************************************************************************************************/
#include <diopi/diopirt.h>

#include "log.h"
#include "ops.h"

namespace impl {
namespace topsrider {

DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores,
                                    double iou_threshold, int64_t offset) {
    TOPSOP_LOG();
    return impl::tops::topsNms(ctx, out, dets, scores, iou_threshold, offset);
}

}  // namespace topsrider
}  // namespace impl
