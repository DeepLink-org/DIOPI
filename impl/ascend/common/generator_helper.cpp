/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>

#include "acloprunner.hpp"

namespace impl {
namespace ascend {

static const size_t seed_size = sizeof(uint64_t);
static const size_t offset_size = sizeof(int64_t);
static const size_t total_size = seed_size + offset_size;

std::pair<uint64_t, int64_t> getSeedAndOffset(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, uint64_t inc) {
    diopiTensorHandle_t stateHandle = nullptr;
    diopiGeneratorGetState(ctx, gen, &stateHandle);
    void* statePtr = nullptr;
    diopiGetTensorData(stateHandle, &statePtr);
    uint64_t currentSeedValue = 0;
    int64_t offsetValue = 0;
    memcpy(&currentSeedValue, statePtr, seed_size);
    memcpy(&offsetValue, statePtr + seed_size, offset_size);

    // update offset
    inc = ((inc + 3) / 4) * 4;
    uint64_t updateOffset = offsetValue + inc;
    memcpy(statePtr + seed_size, &updateOffset, offset_size);
    diopiGeneratorSetState(gen, stateHandle);

    return std::make_pair(currentSeedValue, offsetValue);
}
}  // namespace ascend
}  // namespace impl
