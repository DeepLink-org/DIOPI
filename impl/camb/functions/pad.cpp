/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {
namespace {

std::vector<int> getDim(DiopiTensor tensor) {
    int shapeSize = tensor.shape().size();
    std::vector<int> dim;
    dim.reserve(shapeSize);
    for (int i = 0; i < shapeSize; i++) {
        dim.push_back(static_cast<int>(tensor.shape()[i]));
    }
    if (shapeSize == 3) {
        dim.insert(dim.begin(), 1);
    }
    return dim;
}

}  // namespace

extern "C" {

DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode,
                                const double* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor inputTensorTmp = inputTensor;
    DiopiTensor outTensorTmp = outTensor;

    if (inputTensor.dtype() == diopi_dtype_float64) {
        std::vector<DiopiTensor*> pTensors{&inputTensorTmp};
        DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float32}));
        inputTensorTmp = *pTensors[0];
        DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));
    }

    std::vector<int> padVec(pad.data, pad.data + pad.len);
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc outDesc;
    std::string padMode(mode);
    if (padMode == "constant") {
        inputDesc.set(inputTensorTmp, CNNL_LAYOUT_ARRAY);
        outDesc.set(outTensorTmp, CNNL_LAYOUT_ARRAY);

        bool allPadsIsZero = true;
        for (const auto& i : padVec) {
            if (i != 0) {
                allPadsIsZero = false;
                break;
            }
        }
        if (allPadsIsZero) {
            DIOPI_CALLCNNL(cnnlCopy(handle, inputDesc.get(), inputTensorTmp.data(), outDesc.get(), outTensorTmp.data()));
        }

        auto inputSizes = inputTensor.shape();
        auto lInp = inputTensor.shape().size();
        auto lPad = padVec.size() / 2;
        auto lDiff = lInp - lPad;

        std::vector<int64_t> newShape;
        // for MLU pad
        int newPad[lInp][2], newPadTrans[lInp][2];
        for (size_t i = 0; i < (size_t)lDiff; i++) {
            newShape.emplace_back(inputSizes[i]);
            newPad[i][0] = newPad[i][1] = 0;
        }

        for (size_t i = 0; i < (size_t)lPad; i++) {
            auto padIdx = padVec.size() - ((i + 1) * 2);
            auto newDim = inputSizes[lDiff + i] + padVec[padIdx] + padVec[padIdx + 1];
            newShape.emplace_back(newDim);
            newPad[lDiff + i][0] = padVec[padIdx];
            newPad[lDiff + i][1] = padVec[padIdx + 1];
        }

        void* valuePtr;
        bool tempBool = false;
        int8_t tempI8 = 0;
        uint8_t tempU8 = 0;
        int16_t tempI16 = 0;
        uint16_t tempU16 = 0;
        int32_t tempI32 = 0;
        uint32_t tempU32 = 0;
        int64_t tempI64 = 0;
        uint64_t tempU64 = 0;
        half_float::half tempF16 = static_cast<half_float::half>(0);
        float tempF32 = 0;

        if (value != nullptr) {
            switch (inputTensorTmp.dtype()) {
                case diopi_dtype_bool: {
                    tempBool = static_cast<bool>(*value);
                    valuePtr = &tempBool;
                    break;
                }
                case diopi_dtype_int8: {
                    tempI8 = int8_t(*value);
                    valuePtr = &tempI8;
                    break;
                }
                case diopi_dtype_uint8: {
                    tempU8 = uint8_t(*value);
                    valuePtr = &tempU8;
                    break;
                }
                case diopi_dtype_int16: {
                    tempI16 = int16_t(*value);
                    valuePtr = &tempI16;
                    break;
                }
                case diopi_dtype_uint16: {
                    tempU16 = uint16_t(*value);
                    valuePtr = &tempU16;
                    break;
                }
                case diopi_dtype_int32: {
                    tempI32 = int32_t(*value);
                    valuePtr = &tempI32;
                    break;
                }
                case diopi_dtype_uint32: {
                    tempU32 = uint32_t(*value);
                    valuePtr = &tempU32;
                    break;
                }
                case diopi_dtype_int64: {
                    tempI64 = int64_t(*value);
                    valuePtr = &tempI64;
                    break;
                }
                case diopi_dtype_uint64: {
                    tempU64 = uint64_t(*value);
                    valuePtr = &tempU64;
                    break;
                }
                case diopi_dtype_float16: {
                    tempF16 = half_float::half(*value);
                    valuePtr = &tempF16;
                    break;
                }
                case diopi_dtype_float32: {
                    tempF32 = static_cast<float>(*value);
                    valuePtr = &tempF32;
                    break;
                }
                default:
                    break;
            }
        }
        DIOPI_CALLCNNL(
            cnnlPad(handle, inputDesc.get(), inputTensorTmp.data(), newPad, (value == nullptr) ? nullptr : valuePtr, outDesc.get(), outTensorTmp.data()));
    } else if (padMode == "reflect") {
        std::vector<int> inputDim = getDim(inputTensorTmp);
        std::vector<int> outDim = getDim(outTensorTmp);
        inputDesc.set(inputTensorTmp, CNNL_LAYOUT_NCHW, inputDim);
        outDesc.set(outTensorTmp, CNNL_LAYOUT_NCHW, outDim);
        int padTmp[4];
        if (padVec.size() == 4) {
            for (int i = 0; i < 4; i++) {
                padTmp[i] = static_cast<int>(padVec[i]);
            }
        } else if (padVec.size() == 2) {
            padTmp[2] = padTmp[3] = 0;
            for (int i = 0; i < 2; i++) {
                padTmp[i] = static_cast<int>(padVec[i]);
            }
        } else {
            DIOPI_CHECK(false, "Only supports 2D padding for reflection padding mode now.");
        }
        DIOPI_CALLCNNL(cnnlReflectionPad2d(handle, inputDesc.get(), inputTensorTmp.data(), padTmp, outDesc.get(), outTensorTmp.data()));
    } else if (padMode == "replicate") {
        std::vector<int> inputDim = getDim(inputTensorTmp);
        std::vector<int> outDim = getDim(outTensorTmp);
        inputDesc.set(inputTensorTmp, CNNL_LAYOUT_NCHW, inputDim);
        outDesc.set(outTensorTmp, CNNL_LAYOUT_NCHW, outDim);
        int padTmp[4];
        if (padVec.size() == 4) {
            for (int i = 0; i < 4; i++) {
                padTmp[i] = static_cast<int>(padVec[i]);
            }
        } else if (padVec.size() == 2) {
            padTmp[2] = padTmp[3] = 0;
            for (int i = 0; i < 2; i++) {
                padTmp[i] = static_cast<int>(padVec[i]);
            }
        } else {
            DIOPI_CHECK(false, "Only supports 2D padding for replicate padding mode now.");
        }
        DIOPI_CALLCNNL(cnnlReplicationPad2d(handle, inputDesc.get(), inputTensorTmp.data(), padTmp, outDesc.get(), outTensorTmp.data()));
    } else if (padMode == "circular") {
        inputDesc.set(inputTensorTmp, CNNL_LAYOUT_ARRAY);
        outDesc.set(outTensorTmp, CNNL_LAYOUT_ARRAY);

        auto createSliceOut = [&](auto& dst, auto src, int value, int dim) {
            std::vector<int64_t> sliceShape1(src.shape().size());
            for (int i = 0; i < src.shape().size(); i++) {
                sliceShape1[i] = src.shape()[i];
            }
            sliceShape1[dim] = value;
            diopiSize_t sliceShape(sliceShape1.data(), sliceShape1.size());
            DIOPI_CALL(diopiRequireTensor(ctx, &dst, &sliceShape, nullptr, src.dtype(), diopi_device));
            return diopiSuccess;
        };

        auto sliceConcat1 = [&](auto& dst, auto src, int start, int end, int dim) {
            diopiTensorHandle_t inputSlice = nullptr;
            auto dimValue = end - start;
            DIOPI_CALL(createSliceOut(inputSlice, src, dimValue, dim));
            DIOPI_CALL(diopiSlice(ctx, inputSlice, static_cast<diopiTensorHandle_t>(src), dim, start, end, 1));
            diopiConstTensorHandle_t tensorsCat[2];
            tensorsCat[0] = static_cast<diopiConstTensorHandle_t>(src);
            tensorsCat[1] = static_cast<diopiConstTensorHandle_t>(inputSlice);
            DIOPI_CALL(createSliceOut(dst, src, src.shape()[dim] + dimValue, dim));
            DIOPI_CALL(diopiCat(ctx, dst, tensorsCat, 2, dim));
            return diopiSuccess;
        };

        auto sliceConcat2 = [&](auto& dst, auto src, int start, int end, int dim) {
            diopiTensorHandle_t inputSlice = nullptr;
            auto dimValue = end - start;
            DIOPI_CALL(createSliceOut(inputSlice, src, dimValue, dim));
            DIOPI_CALL(diopiSlice(ctx, inputSlice, static_cast<diopiTensorHandle_t>(src), dim, start, end, 1));
            diopiConstTensorHandle_t tensorsCat[2];
            tensorsCat[0] = static_cast<diopiConstTensorHandle_t>(inputSlice);
            tensorsCat[1] = static_cast<diopiConstTensorHandle_t>(src);
            DIOPI_CALL(createSliceOut(dst, src, src.shape()[dim] + dimValue, dim));
            DIOPI_CALL(diopiCat(ctx, dst, tensorsCat, 2, dim));
            return diopiSuccess;
        };

        diopiTensorHandle_t catOut1 = nullptr;
        DIOPI_CALL(sliceConcat1(catOut1, inputTensorTmp, 0, padVec[padVec.size() - 1], 2));

        DiopiTensor catOut1Tensor(catOut1);
        diopiTensorHandle_t catOut2 = nullptr;
        DIOPI_CALL(sliceConcat2(catOut2, catOut1Tensor, -(padVec[padVec.size() - 1] + padVec[padVec.size() - 2]), -padVec[padVec.size() - 1], 2));
        if (padVec.size() <= 2) {
            DIOPI_CALL(diopiCopyInp(ctx, catOut2, static_cast<diopiTensorHandle_t>(outTensorTmp)));
        }

        if (padVec.size() > 2) {
            DiopiTensor catOut2Tensor(catOut2);
            diopiTensorHandle_t catOut3 = nullptr;
            DIOPI_CALL(sliceConcat1(catOut3, catOut2Tensor, 0, padVec[padVec.size() - 3], 3));

            DiopiTensor catOut3Tensor(catOut3);
            diopiTensorHandle_t catOut4 = nullptr;
            DIOPI_CALL(sliceConcat2(catOut4, catOut3Tensor, -(padVec[padVec.size() - 3] + padVec[padVec.size() - 4]), -padVec[padVec.size() - 3], 3));
            DIOPI_CALL(diopiCopyInp(ctx, catOut4, static_cast<diopiTensorHandle_t>(outTensorTmp)));
        }
    } else {
        DIOPI_CHECK(false, "Only supports constant, reflect, circular and replicate now.");
    }
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
