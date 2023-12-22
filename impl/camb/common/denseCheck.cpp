/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

bool denseCheck(const DiopiTensor& src) {
    int dim = src.dim();
    std::vector<int> stride(dim, 1);
    std::vector<int> shape(dim, 1);

    for (int i = 0; i < dim; i++) {
        stride[i] = src.stride()[i];
        shape[i] = src.shape()[i];

        if (src.stride()[i] == 0 || src.shape()[i] == 0) {
            return false;
        }
    }

    std::sort(stride.begin(), stride.end());

    // e.g. shape = 2,3,4,5,stride = 1,3,12,60
    if (stride[0] != 1) {
        return false;
    }
    int cur = 1;
    for (int i = 1; i < dim; i++) {
        cur = stride[i] / stride[i - 1];
        if (std::find(shape.begin(), shape.end(), cur) != shape.end()) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

}  // namespace camb
}  // namespace impl
