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

DIOPI_API diopiError_t diopiInputIdsEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t input_ids,
                                                               diopiConstTensorHandle_t embedding_table, const int64_t input_lengths,
                                                               const int64_t hidden_units) {
    if (from_tensor == nullptr || input_ids == nullptr || embedding_table == nullptr) {
        return diopiErrorOccurred;
    }

    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, input_ids);
    return diopiSuccess;
}

}  // extern "C"
