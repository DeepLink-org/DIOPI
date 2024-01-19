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

bool isSlice(const DiopiTensor& src) {
    int dim = src.dim();
    std::vector<std::pair<int64_t, int64_t>> stridesSizes(dim, std::pair<int64_t, int64_t>(1, 1));

    for (int i = 0; i < dim; i++) {
        stridesSizes[i] = std::pair<int64_t, int64_t>(src.stride()[i], src.shape()[i]);

        if (src.stride()[i] == 0 || src.shape()[i] == 0) {
            return false;
        }
    }

    sort(stridesSizes.begin(), stridesSizes.end(), [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) { return a.first < b.first; });

    int cur = 1;

    // case1 input: sizes: [128, 12, 64, 197], stride: [453888, 64, 1, 2304] should pass
    for (int i = 0; i < dim; i++) {
        if (stridesSizes[i].first % cur != 0) {
            return false;
        }
        cur *= stridesSizes[i].second;
    }

    return true;
}

bool isSparse(const DiopiTensor& src) {
    int dim = src.dim();
    std::vector<std::pair<int64_t, int64_t>> stridesSizes(dim, std::pair<int64_t, int64_t>(1, 1));

    for (int i = 0; i < dim; i++) {
        stridesSizes[i] = std::pair<int64_t, int64_t>(src.stride()[i], src.shape()[i]);

        if (src.stride()[i] == 0 || src.shape()[i] == 0) {
            return false;
        }
    }

    sort(stridesSizes.begin(), stridesSizes.end(), [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) { return a.first < b.first; });

    // sizes: [128, 768, 14, 14], stride: [151296, 1, 10752, 768]
    int cur = 1;

    for (int i = 0; i < dim; i++) {
        if (stridesSizes[i].first < cur) {
            return false;
        }
        cur = stridesSizes[i].second * stridesSizes[i].first;
    }

    return true;
}

diopiError_t getDenseStride(const DiopiTensor& src, std::vector<int64_t>& dstStride) {
    int64_t dim = src.dim();
    std::vector<std::pair<int64_t, int64_t>> stridesSizes(dim, std::pair<int64_t, int64_t>(1, 1));
    for (int64_t i = 0; i < dim; i++) {
        stridesSizes[i] = std::pair<int64_t, int64_t>(src.stride()[i], i);
    }
    sort(stridesSizes.begin(), stridesSizes.end(), [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) { return a.first < b.first; });

    // shape 128 12 64 197 stride 453888 64 1 2304 index 0 1 2 3-> sorted  (1,2) (64,1) (2304,3) (453888,0)
    int64_t curStride = 1;
    for (int64_t i = 0; i < dim; i++) {
        dstStride[stridesSizes[i].second] = curStride;
        curStride *= src.shape()[stridesSizes[i].second];
    }
    return diopiSuccess;
}

diopiError_t sliceToDense(diopiContextHandle_t ctx, DiopiTensor& src, DiopiTensor& dst) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    int dim = src.dim();

    // real target shape and stride
    std::vector<int64_t> targetShape = src.shape();
    std::vector<int64_t> targetStride = dst.stride();
    // std::vector<int64_t> targetStride(dim,0);
    // getDenseStride(src, targetStride);

    std::vector<std::pair<int64_t, int64_t>> stridesSizes(dim, std::pair<int64_t, int64_t>(1, 1));
    for (int i = 0; i < dim; i++) {
        stridesSizes[i] = std::pair<int, int>(src.stride()[i], src.shape()[i]);
    }
    sort(stridesSizes.begin(), stridesSizes.end(), [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) { return a.first > b.first; });

    // 得到按照stride从大到小顺序排列的shape，实测这种copy最快，也可以考虑用slice;
    // e.g. shape 128 12 64 197 stride 453888 64 1 2304-> shape 128 197 64 12 stride 453888 2304 64 1
    // e.g. shape: [128, 768, 14, 14], stride: [151296, 1, 10752, 768]
    // -> shape: 128 14 14 768, stride:151296, 10752, 768, 1
    std::vector<int64_t> generateShape;
    std::vector<int64_t> generateStride;
    std::vector<int64_t> generateOutStride(dim, 0);
    for (int i = 0; i < dim; i++) {
        generateShape.push_back(stridesSizes[i].second);
        generateStride.push_back(stridesSizes[i].first);
    }

    int64_t curOutStride = 1;
    for (int i = 0; i < dim; i++) {
        generateOutStride[dim - 1 - i] = curOutStride;
        curOutStride *= generateShape[dim - 1 - i];
    }

    CnnlTensorDesc srcTensorDesc(src, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc dstTensorDesc(dst, CNNL_LAYOUT_ARRAY);
    srcTensorDesc.set(src.dtype(), generateShape, generateStride, CNNL_LAYOUT_ARRAY);
    dstTensorDesc.set(dst.dtype(), generateShape, generateOutStride, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlCopy(handle, srcTensorDesc.get(), src.data(), dstTensorDesc.get(), dst.data()));
    dst.asStrided(targetShape, targetStride);
    return diopiSuccess;
}

diopiError_t toDense(diopiContextHandle_t ctx, DiopiTensor& src, DiopiTensor& dst) {
    if (isSlice(src)) {
        std::vector<int64_t> targetStride(src.dim(), 0);
        getDenseStride(src, targetStride);
        dst = requiresTensor(ctx, src.shape(), targetStride, src.dtype());
        sliceToDense(ctx, src, dst);
    } else if (isSparse(src)) {
        std::vector<int64_t> targetStride(src.dim(), 0);
        getDenseStride(src, targetStride);
        dst = requiresTensor(ctx, src.shape(), targetStride, src.dtype());
        sliceToDense(ctx, src, dst);
    } else {
        // for some special cases(broadcast), we set it as contiguous copy, but can be modified in the future
        dst = requiresTensor(ctx, src.shape(), src.dtype(), diopiMemoryFormat_t::Contiguous);
        CnnlTensorDesc srcTensorDesc(src, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc dstTensorDesc(dst, CNNL_LAYOUT_ARRAY);
        cnnlHandle_t handle = cnnlHandlePool.get(ctx);
        DIOPI_CALL_CNNL(cnnlCopy(handle, srcTensorDesc.get(), src.data(), dstTensorDesc.get(), dst.data()));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
