/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
// infer the size of a expanded tensor based on the sizes of two input tensors a and b
static std::vector<int64_t> inferSize(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    int32_t dimsA = a.size();
    int32_t dimsB = b.size();
    int32_t ndim = dimsA > dimsB ? dimsA : dimsB;
    std::vector<int64_t> expandedSize(ndim);
    for (auto i = ndim - 1; i >= 0; --i) {
        auto offset = ndim - 1 - i;
        auto dimA = dimsA - 1 - offset;
        auto dimB = dimsB - 1 - offset;
        auto sizeA = (dimA >= 0) ? a[dimA] : 1;
        auto sizeB = (dimB >= 0) ? b[dimB] : 1;
        assert((sizeA == sizeB || sizeA == 1 || sizeB == 1) && "The size of tensor a must match the size of tensor b at a non-singleton dimension");
        expandedSize[i] = sizeA == 1 ? sizeB : sizeA;
    }
    return expandedSize;
}

// to satisfy the kernel requirement, the bool index tensor in the input indices list is converted to an equivalent int index tensor
static diopiError_t indexPreProcess(diopiContextHandle_t ctx, DiopiTensor inputTensor, std::vector<DiopiTensor> indicesTensors,
                                    std::vector<DiopiTensor>& indicesTensorsCast) {
    // expand bool tensor or byte tensor into 1 or more int tensors
    bool boolTensorConvertToEmptyTensor = false;
    for (auto indexTensor : indicesTensors) {
        if (!indexTensor.defined()) {
            indicesTensorsCast.emplace_back();
        } else {
            if (indexTensor.dtype() == diopi_dtype_uint8 || indexTensor.dtype() == diopi_dtype_bool) {
                // the sizes of byte tensor or bool tensor must match the sizes of the corresponding dimensions in input
                for (auto j = 0; j < indexTensor.dim(); ++j) {
                    int64_t srcIdx = indicesTensorsCast.size() + j;
                    if (indexTensor.size(j) != inputTensor.size(srcIdx)) {
                        DIOPI_CHECK(false, "Invalid mask");
                    }
                }
                // replace with nonzeros
                diopiTensorHandle_t out = nullptr;
                DIOPI_CALL(diopiNonzero(ctx, &out, indexTensor.tensorHandle()));
                DiopiTensor nonzeroTensor(out);
                // empty tensor judgment
                if (nonzeroTensor.numel()) {
                    for (auto j = 0; j < indexTensor.dim(); ++j) {
                        std::vector<int64_t> selectShape{nonzeroTensor.shape()[0]};
                        DiopiTensor selectTensor = requiresTensor(ctx, selectShape, diopi_dtype_int32);
                        DIOPI_CALL(diopiSelect(ctx, selectTensor.tensorHandle(), nonzeroTensor.tensorHandle(), 1, j));
                        indicesTensorsCast.emplace_back(std::move(selectTensor));
                    }
                } else {
                    // specical case: bool tensor -> empty int tensor
                    for (auto j = 0; j < indexTensor.dim(); ++j) {
                        std::vector<int64_t> emptyShape{0};
                        DiopiTensor emptyTensor = requiresTensor(ctx, emptyShape, diopi_dtype_int32);
                        indicesTensorsCast.emplace_back(std::move(emptyTensor));
                    }
                    boolTensorConvertToEmptyTensor = true;
                }
            } else {
                // int tensor
                indicesTensorsCast.emplace_back(std::move(indexTensor));
            }
        }
    }
    indicesTensorsCast.resize(indicesTensors.size());

    // handle the broadcast of special empty index tensor
    if (boolTensorConvertToEmptyTensor) {
        bool first = true;
        std::vector<int64_t> sizes;
        for (auto indexTensorCast : indicesTensorsCast) {
            if (!indexTensorCast.defined()) {
                continue;
            } else if (first) {
                sizes = indexTensorCast.shape();
                first = false;
            } else {
                sizes = inferSize(sizes, indexTensorCast.shape());
            }
        }
        for (auto& indexTensorCast : indicesTensorsCast) {
            if (indexTensorCast.defined() && !indexTensorCast.numel()) {
                std::vector<int64_t> emptyExpandShape(sizes.size(), 1);
                emptyExpandShape.insert(emptyExpandShape.begin(), 0);
                indexTensorCast.view(emptyExpandShape);
            }
        }
    }
    return diopiSuccess;
}

static diopiError_t indexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                             diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor valuesTensor(values);
    DiopiTensor outputTensor(out);

    DIOPI_CHECK(indicesCounts <= inputTensor.dim(), "Too many indices for tensor of dimension");
    DIOPI_CHECK(inputTensor.dtype() == valuesTensor.dtype(), "Input and values must have the same dtype");

    std::vector<DiopiTensor> indicesTensors;
    for (auto i = 0; i < indicesCounts; ++i) {
        DiopiTensor indexTensor(indices[i]);
        if (!indexTensor.defined()) {
            indicesTensors.emplace_back();
        } else {
            if (indexTensor.dtype() == diopi_dtype_int64) {
                DIOPI_CALL(dataTypeCast(ctx, indexTensor, diopi_dtype_int32));
            }
            DIOPI_CHECK(indexTensor.dtype() == diopi_dtype_int32 || indexTensor.dtype() == diopi_dtype_bool || indexTensor.dtype() == diopi_dtype_uint8,
                        "Tensors used as indices must be int32, bool or uint8 tensors");
            // expand bool tensor (masks) or byte tensor (masks) into 1 or more int tensors
            if (indexTensor.dtype() == diopi_dtype_uint8 || indexTensor.dtype() == diopi_dtype_bool) {
                // the sizes of byte tensor or bool tensor must match the sizes of the corresponding dimensions in input
                for (auto j = 0; j < indexTensor.dim(); ++j) {
                    int64_t srcIdx = indicesTensors.size() + j;
                    if (indexTensor.size(j) != inputTensor.size(srcIdx)) {
                        DIOPI_CHECK(false, "Invalid mask");
                    }
                }
                // replace with nonzeros
                diopiTensorHandle_t nonzeros0ut = nullptr;
                DIOPI_CALL(diopiNonzero(ctx, &nonzeros0ut, indexTensor.tensorHandle()));
                DiopiTensor nonzeroTensor(nonzeros0ut);
                // empty tensor judgment
                if (nonzeroTensor.numel()) {
                    for (auto j = 0; j < indexTensor.dim(); ++j) {
                        std::vector<int64_t> selectShape{nonzeroTensor.shape()[0]};
                        DiopiTensor selectTensor = requiresTensor(ctx, selectShape, diopi_dtype_int32);
                        DIOPI_CALL(diopiSelect(ctx, selectTensor.tensorHandle(), nonzeroTensor.tensorHandle(), 1, j));
                        indicesTensors.emplace_back(std::move(selectTensor));
                    }
                } else {
                    // specical case: bool tensor -> empty int tensor
                    diopiScalar_t value = constructDiopiScalarT(inputTensor.dtype(), 0);
                    DIOPI_CALL(diopiFill(ctx, out, &value));
                    return diopiSuccess;
                }
            } else {
                // int tensor
                if (!indexTensor.numel()) {
                    diopiScalar_t value = constructDiopiScalarT(inputTensor.dtype(), 0);
                    DIOPI_CALL(diopiFill(ctx, out, &value));
                    return diopiSuccess;
                }
                indicesTensors.emplace_back(indexTensor);
            }
        }
    }

    const int64_t arraySize = indicesTensors.size();
    std::vector<CnnlTensorDesc> indicesDesc(arraySize);
    std::vector<cnnlTensorDescriptor_t> indicesDescT(arraySize);
    std::vector<void*> indicesPtrList(arraySize);
    for (auto i = 0; i < arraySize; ++i) {
        if (indicesTensors[i].defined()) {
            indicesDesc[i].set(indicesTensors[i], CNNL_LAYOUT_ARRAY);
            indicesDescT[i] = indicesDesc[i].get();
            indicesPtrList[i] = indicesTensors[i].data();
        }
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc valuesDesc(valuesTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(
        cnnlGetIndexPutWorkspaceSize(handle, inputDesc.get(), indicesDescT.data(), indicesDescT.size(), valuesDesc.get(), accumulate, &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlIndexPut(handle,
                                inputDesc.get(),
                                inputTensor.data(),
                                indicesDescT.data(),
                                indicesPtrList.data(),
                                indicesDescT.size(),
                                valuesDesc.get(),
                                valuesTensor.data(),
                                workspace,
                                workspaceSize,
                                accumulate,
                                true,
                                outputDesc.get(),
                                outputTensor.data()));
    return diopiSuccess;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    const int64_t arraySize = 8;

    DiopiTensor inputTensor(input);
    DiopiTensor inputTensorTmp = inputTensor;
    DIOPI_CHECK(nums <= inputTensor.dim(), "Too many indices for tensor of dimension");
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, {&inputTensorTmp}, supportedDtypes));
    std::vector<DiopiTensor> indicesTensors(arraySize);
    for (auto i = 0; i < nums; ++i) {
        DiopiTensor indexTensor(indices[i]);
        if (indexTensor.defined()) {
            if (indexTensor.dtype() == diopi_dtype_int64) {
                DIOPI_CALL(dataTypeCast(ctx, indexTensor, diopi_dtype_int32));
            }
            DIOPI_CHECK(indexTensor.dtype() == diopi_dtype_int32 || indexTensor.dtype() == diopi_dtype_bool || indexTensor.dtype() == diopi_dtype_uint8,
                        "Tensors used as indices must be int32, bool or uint8 tensors");
            indicesTensors[i] = indexTensor;
        }
    }

    std::vector<DiopiTensor> indicesTensorsCast;
    DIOPI_CALL(indexPreProcess(ctx, inputTensorTmp, indicesTensors, indicesTensorsCast));
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);

    std::vector<CnnlTensorDesc> indicesDesc(arraySize);
    std::vector<cnnlTensorDescriptor_t> indicesDescT(arraySize);
    std::vector<void*> indicesPtrList(arraySize);
    for (auto i = 0; i < arraySize; ++i) {
        if (indicesTensorsCast[i].defined()) {
            indicesDesc[i].set(indicesTensorsCast[i], CNNL_LAYOUT_ARRAY);
            indicesDescT[i] = indicesDesc[i].get();
            indicesPtrList[i] = indicesTensorsCast[i].data();
        }
    }

    int32_t outputDescDim = 0;
    std::vector<int32_t> outputDescDims(arraySize);
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexOutputDim(handle, inputDesc.get(), indicesDescT.data(), &outputDescDim, outputDescDims.data()));
    outputDescDims.resize(outputDescDim);

    std::vector<int64_t> outTensorShape(outputDescDims.begin(), outputDescDims.end());
    DiopiTensor outTensor = requiresTensor(ctx, outTensorShape, inputTensorTmp.dtype());
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor outputDims = requiresTensor(ctx, {static_cast<int64_t>(outTensorShape.size())}, diopi_dtype_int64);
    cnrtMemcpy(outputDims.data(), outTensorShape.data(), sizeof(int64_t) * outTensorShape.size(), cnrtMemcpyHostToDev);
    DiopiTensor outputDim = requiresTensor(ctx, {1}, diopi_dtype_int32);
    cnrtMemcpy(outputDim.data(), &outputDescDim, sizeof(int32_t), cnrtMemcpyHostToDev);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexWorkspaceSize(handle, inputDesc.get(), indicesDescT.data(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlAdvancedIndex(handle,
                                     inputDesc.get(),
                                     inputTensorTmp.data(),
                                     indicesDescT.data(),
                                     indicesPtrList.data(),
                                     workspace,
                                     workspaceSize,
                                     outDesc.get(),
                                     outTensor.data(),
                                     outputDims.data(),
                                     outputDim.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, inputTensor.dtype()));
    *out = diopiTensorHandle_t(outTensor);
    return diopiSuccess;
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t zerosLikeInput, diopiConstTensorHandle_t* indices,
                                int64_t nums, diopiConstTensorHandle_t gradOutput) {
    DIOPI_CALL(indexPut(ctx, gradInput, zerosLikeInput, gradOutput, indices, nums, true));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
