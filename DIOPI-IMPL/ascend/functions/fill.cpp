#include <diopi/functions.h>
#include "../common/acloprunner.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
