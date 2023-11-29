/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>

#include <iostream>
#include <vector>

extern "C" {

DIOPI_API diopiError_t diopiEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t embedding_table,
                                                       diopiConstTensorHandle_t all_ids, const int64_t batch_size, const int64_t hidden_units,
                                                       const int64_t step) {
    if (from_tensor == nullptr || embedding_table == nullptr || all_ids == nullptr) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;

    diopiGetTensorDtype(all_ids, &in_type);
    diopiGetTensorShape(all_ids, &in_shape);
    diopiGetTensorStride(all_ids, &in_stride);
    diopiGetTensorDevice(all_ids, &in_device);

    diopiTensorHandle_t this_step_ids;
    diopiSize_t this_step_shape;
    this_step_shape.len = 1;
    this_step_shape.data = &batch_size;
    diopiSize_t this_step_stride;
    this_step_stride.len = 1;
    this_step_stride.data = in_stride.data + 1;
    diopiRequireTensor(ctx, &this_step_ids, &this_step_shape, &this_step_stride, in_type, in_device);

    diopiSelect(ctx, this_step_ids, all_ids, 0, step);
    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, this_step_ids);
    return diopiSuccess;
}

}  // extern "C"

