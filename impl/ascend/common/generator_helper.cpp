/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>

#include "acloprunner.hpp"
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

static const size_t seedSize = sizeof(uint64_t);
static const size_t offsetSize = sizeof(int64_t);

std::pair<uint64_t, int64_t> getSeedAndOffset(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, uint64_t inc) {
    diopiTensorHandle_t stateHandle = nullptr;
    diopiGeneratorGetState(ctx, gen, &stateHandle);
    void* statePtr = nullptr;
    diopiGetTensorData(stateHandle, &statePtr);
    uint64_t currentSeedValue = 0;
    int64_t offsetValue = 0;
    memcpy(&currentSeedValue, statePtr, seedSize);
    memcpy(&offsetValue, static_cast<char*>(statePtr) + seedSize, offsetSize);

    // update offset
    inc = ((inc + 3) / 4) * 4;
    uint64_t updateOffset = offsetValue + inc;
    memcpy(static_cast<char*>(statePtr) + seedSize, &updateOffset, offsetSize);
    diopiGeneratorSetState(gen, stateHandle);

    return std::make_pair(currentSeedValue, offsetValue);
}
}  // namespace ascend
}  // namespace impl
