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
    std::vector<std::pair<int, int>> stridesSizes(dim, std::pair<int, int>(1, 1));

    for (int i = 0; i < dim; i++) {
        stridesSizes[i] = std::pair<int, int>(src.stride()[i], src.shape()[i]);

        if (src.stride()[i] == 0 || src.shape()[i] == 0) {
            return false;
        }
    }

    sort(stridesSizes.begin(), stridesSizes.end(), [](std::pair<int, int> a, std::pair<int, int> b) { return a.first < b.first; });

    // e.g. shape = 2,3,4,5,stride = 1,2,6,24 pass
    // e.g. shape = 2,3,4,5, stride = 1,2,6,12 should not pass
    int cur = 1;
    for (int i = 0; i < dim; i++) {
        if (stridesSizes[i].first != cur) {
            return false;
        }
        cur *= stridesSizes[i].second;
    }

    return true;
}

}  // namespace camb
}  // namespace impl
