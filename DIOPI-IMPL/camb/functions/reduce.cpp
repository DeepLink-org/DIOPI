/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <algorithm>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

/*
Get real sorted dim
eg: getRealDims({-1,1,2,3,-2}, 5) -> {1,2,3,4}
*/
std::vector<int64_t> getRealDims(std::vector<int64_t> inputDim, int64_t tDim) {
    // handle negative dims
    for (int64_t& i : inputDim) {
        if (i < 0) {
            i = i + tDim;
        }
    }
    // remove duplicate dims and sort them
    std::vector<int64_t> dimVec(inputDim);
    std::set<int64_t> s(dimVec.begin(), dimVec.end());
    dimVec.assign(s.begin(), s.end());
    return dimVec;
}

std::vector<int> inferDescShape(std::vector<int64_t> inputDim, std::vector<int64_t> reduceDim, bool keepdim) {
    std::vector<int> outputDim(inputDim.begin(), inputDim.end());
    if (inputDim.empty()) {
        return outputDim;
    }
    int num = 0;
    for (auto i : reduceDim) {
        if (keepdim) {
            outputDim[i] = 1;
        } else {
            auto it = outputDim.begin() + i - num;
            outputDim.erase(it);
            num++;
        }
    }
    return outputDim;
}
struct HashCnnlReduceOp {
    int64_t operator()(const cnnlReduceOp_t& reduceOp) const { return static_cast<int64_t>(reduceOp); }
};
static std::unordered_map<cnnlReduceOp_t, std::set<diopiDtype_t>, HashCnnlReduceOp> supportedTypeTable = {
    {CNNL_REDUCE_ADD, {diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_AVG, {diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_MUL, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_MAX, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_MIN, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_AND, {diopi_dtype_bool, diopi_dtype_uint8, diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_OR, {diopi_dtype_bool, diopi_dtype_uint8, diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_NORM1, {diopi_dtype_float16, diopi_dtype_float32}},
    {CNNL_REDUCE_NORM2, {diopi_dtype_float16, diopi_dtype_float32}}};

diopiError_t reduceInternal(diopiContextHandle_t ctx, DiopiTensor& inputTr, DiopiTensor& outputTr, DiopiTensor& indexTr, const std::vector<int64_t> reduceDim,
                            cnnlReduceOp_t reduceOp) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DIOPI_CHECK(inputTr.isContiguous(), "input tensor should be contiguous");

    CnnlReduceDescriptor reduceDesc;
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc outputDesc;
    CnnlTensorDesc indexDesc;

    cnnlDataType_t cnnlDtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&cnnlDtype, inputTr.dtype()));
    if (reduceOp == CNNL_REDUCE_AVG && inputTr.dtype() == diopi_dtype_float16) {
        cnnlDtype = CNNL_DTYPE_FLOAT;
    }

    // Only Min and Max Ops have indices as result.when reduce_dim > 1,
    auto reduceIndices =
        ((reduceOp == CNNL_REDUCE_MAX || reduceOp == CNNL_REDUCE_MIN) && !reduceDim.empty()) ? CNNL_REDUCE_FLATTENED_INDICES : CNNL_REDUCE_NO_INDICES;

    if (reduceDim.empty() || reduceDim.size() == inputTr.dim() + 1) {
        /* FULL-REDUCE: axis = [-1] instead of [0, 1, 2, ..., n] */
        std::vector<int64_t> fullReduce(1, -1);
        std::vector<int64_t> fakeSize(inputTr.dim(), 1);
        reduceDesc.set(inputTr, fullReduce, reduceOp, reduceIndices, CNNL_32BIT_INDICES, cnnlDtype);
        inputDesc.set(inputTr, CNNL_LAYOUT_ARRAY);
        DiopiTensor fakeTensor = requiresTensor(ctx, fakeSize, outputTr.dtype());
        outputDesc.set(fakeTensor, CNNL_LAYOUT_ARRAY);
        DiopiTensor fakeTensor2 = requiresTensor(ctx, fakeSize, indexTr.dtype());
        // index_desc.set_reduce(fake_tensor2);
        indexDesc.set(fakeTensor2, CNNL_LAYOUT_ARRAY);
    } else {
        reduceDesc.set(inputTr, reduceDim, reduceOp, reduceIndices, CNNL_32BIT_INDICES, cnnlDtype);
        inputDesc.set(inputTr, CNNL_LAYOUT_ARRAY);
        auto descShape = inferDescShape(inputTr.shape(), reduceDim, true);
        outputDesc.set(outputTr, CNNL_LAYOUT_ARRAY, descShape);
        indexDesc.set(indexTr, CNNL_LAYOUT_ARRAY, descShape);
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, inputDesc.get(), outputDesc.get(), reduceDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlReduce(handle,
                              reduceDesc.get(),
                              workspacePtr,
                              workspaceSize,
                              nullptr,
                              inputDesc.get(),
                              inputTr.data(),
                              sizeof(int) * indexTr.numel(),
                              reduceIndices != CNNL_REDUCE_NO_INDICES ? indexTr.data() : nullptr,
                              nullptr,
                              outputDesc.get(),
                              outputTr.data()));

    return diopiSuccess;
}

diopiError_t reduceImpl(diopiContextHandle_t ctx, DiopiTensor& outputTr, DiopiTensor& indexTr, DiopiTensor& inputTr, cnnlReduceOp_t reduceOp) {
    std::vector<int64_t> reduceDim;
    for (int64_t i = 0; i <= inputTr.dim(); i++) {
        reduceDim.push_back(i);
    }
    auto supportedDtypes = supportedTypeTable.find(reduceOp);
    std::vector<DiopiTensor*> pTensors{&inputTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes->second));

    if (outputTr.dtype() != inputTr.dtype()) {
        auto outputTmpTr = requiresTensor(ctx, outputTr.shape(), inputTr.dtype());
        DIOPI_CALL(reduceInternal(ctx, inputTr, outputTmpTr, indexTr, reduceDim, reduceOp));
        DIOPI_CALL(dataTypeCast(ctx, outputTr, outputTmpTr));
    } else {
        DIOPI_CALL(reduceInternal(ctx, inputTr, outputTr, indexTr, reduceDim, reduceOp));
    }
    return diopiSuccess;
}

diopiError_t reduceDimImpl(diopiContextHandle_t ctx, DiopiTensor& outputTr, DiopiTensor& indexTr, DiopiTensor& inputTr, const std::vector<int64_t> dimVec,
                           const bool keepdim, cnnlReduceOp_t reduceOp) {
    std::vector<int64_t> reduceDim = getRealDims(dimVec, inputTr.dim());
    auto supportedDtypes = supportedTypeTable.find(reduceOp);
    std::vector<DiopiTensor*> pTensors{&inputTr};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes->second));

    if (outputTr.dtype() != inputTr.dtype()) {
        auto outputTmpTr = requiresTensor(ctx, outputTr.shape(), inputTr.dtype());
        DIOPI_CALL(reduceInternal(ctx, inputTr, outputTmpTr, indexTr, reduceDim, reduceOp));
        DIOPI_CALL(dataTypeCast(ctx, outputTr, outputTmpTr));
    } else {
        DIOPI_CALL(reduceInternal(ctx, inputTr, outputTr, indexTr, reduceDim, reduceOp));
    }
    return diopiSuccess;
}

extern "C" {

diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);
    auto indexTr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    std::vector<int64_t> dimVec(dim.data, dim.data + dim.len);

    DIOPI_CALL(reduceDimImpl(ctx, outputTr, indexTr, inputTr, dimVec, false, CNNL_REDUCE_ADD));

    return diopiSuccess;
}

diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);
    auto indexTr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    std::vector<int64_t> dimVec(dim.data, dim.data + dim.len);

    DIOPI_CALL(reduceDimImpl(ctx, outputTr, indexTr, inputTr, dimVec, false, CNNL_REDUCE_AVG));
    return diopiSuccess;
}

diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);
    auto indexTr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    DIOPI_CALL(reduceDimImpl(ctx, outputTr, indexTr, inputTr, {*dim}, false, CNNL_REDUCE_MUL));
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t minIndices, diopiConstTensorHandle_t input, int64_t dim) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(min);
    DiopiTensor indexTr(minIndices);
    // Note: camb index out is int32 dtype
    auto indexTmpTr = requiresTensor(ctx, indexTr.shape(), diopi_dtype_int32);

    DIOPI_CALL(reduceDimImpl(ctx, outputTr, indexTmpTr, inputTr, {dim}, false, CNNL_REDUCE_MIN));

    DIOPI_CALL(dataTypeCast(ctx, indexTr, indexTmpTr));
    return diopiSuccess;
}

diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(min);
    auto indexTr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    DIOPI_CALL(reduceImpl(ctx, outputTr, indexTr, inputTr, CNNL_REDUCE_MIN));

    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t maxIndices, diopiConstTensorHandle_t input, int64_t dim) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(max);
    DiopiTensor indexTr(maxIndices);
    auto indexTmpTr = requiresTensor(ctx, indexTr.shape(), diopi_dtype_int32);

    DIOPI_CALL(reduceDimImpl(ctx, outputTr, indexTmpTr, inputTr, {dim}, false, CNNL_REDUCE_MAX));

    DIOPI_CALL(dataTypeCast(ctx, indexTr, indexTmpTr));
    return diopiSuccess;
}

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    DiopiTensor inputTr(input);
    DiopiTensor outputTr(max);
    auto indexTr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    DIOPI_CALL(reduceImpl(ctx, outputTr, indexTr, inputTr, CNNL_REDUCE_MAX));

    return diopiSuccess;
}

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    float norm = p->fval;
    if (DiopiDataType().isInteger(p->stype)) norm = p->ival;
    DIOPI_CHECK(norm == 1.0 || norm == 2.0, "camb only support L1-Norm as p=1.0 and L2-Norm as p=2.0");

    DiopiTensor inputTr(input);
    DiopiTensor outputTr(out);
    auto indexTr = requiresTensor(ctx, {1}, diopi_dtype_int32);

    std::vector<int64_t> dimVec(dim.data, dim.data + dim.len);
    DIOPI_CALL(reduceDimImpl(ctx, outputTr, indexTr, inputTr, dimVec, false, norm == 1.0 ? CNNL_REDUCE_NORM1 : CNNL_REDUCE_NORM2));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
