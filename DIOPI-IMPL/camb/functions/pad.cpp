/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {
namespace {

std::vector<int> getDim(DiopiTensor tensor) {
    int shape_size = tensor.shape().size();
    std::vector<int> dim;
    for (int i = 0; i < shape_size; i++) {
        dim.push_back(static_cast<int>(tensor.shape()[i]));
    }
    if (shape_size == 3) {
        dim.insert(dim.begin(), 1);
    }
    return dim;
}

}  // namespace

extern "C" {

DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode,
                                double* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    DiopiTensor input_tensor_tmp = input_tensor;
    DiopiTensor out_tensor_tmp = out_tensor;

    if (input_tensor.dtype() == diopi_dtype_float64) {
        std::vector<DiopiTensor*> pTensors{&input_tensor_tmp};
        DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_float32}));
        input_tensor_tmp = *pTensors[0];
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));
    }

    std::vector<int> pad_vec(pad.data, pad.data + pad.len);
    CnnlTensorDesc input_desc;
    CnnlTensorDesc out_desc;
    std::string pad_mode(mode);
    if (pad_mode == "constant") {
        input_desc.set(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
        out_desc.set(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

        bool all_pads_is_zero = true;
        for (const auto& i : pad_vec) {
            if (i != 0) {
                all_pads_is_zero = false;
                break;
            }
        }
        if (all_pads_is_zero) {
            DIOPI_CALLCNNL(cnnlCopy(handle, input_desc.get(), input_tensor_tmp.data(), out_desc.get(), out_tensor_tmp.data()));
        }

        auto input_sizes = input_tensor.shape();
        auto l_inp = input_tensor.shape().size();
        auto l_pad = pad_vec.size() / 2;
        auto l_diff = l_inp - l_pad;

        std::vector<int64_t> new_shape;
        // for MLU pad
        int new_pad[l_inp][2], new_pad_trans[l_inp][2];
        for (size_t i = 0; i < (size_t)l_diff; i++) {
            new_shape.emplace_back(input_sizes[i]);
            new_pad[i][0] = new_pad[i][1] = 0;
        }

        for (size_t i = 0; i < (size_t)l_pad; i++) {
            auto pad_idx = pad_vec.size() - ((i + 1) * 2);
            auto new_dim = input_sizes[l_diff + i] + pad_vec[pad_idx] + pad_vec[pad_idx + 1];
            new_shape.emplace_back(new_dim);
            new_pad[l_diff + i][0] = pad_vec[pad_idx];
            new_pad[l_diff + i][1] = pad_vec[pad_idx + 1];
        }

        void* value_ptr;
        bool temp_bool = 0;
        int8_t temp_i8 = 0;
        uint8_t temp_u8 = 0;
        int16_t temp_i16 = 0;
        uint16_t temp_u16 = 0;
        int32_t temp_i32 = 0;
        uint32_t temp_u32 = 0;
        int64_t temp_i64 = 0;
        uint64_t temp_u64 = 0;
        half_float::half temp_f16 = static_cast<half_float::half>(0);
        float temp_f32 = 0;

        if (value != nullptr) {
            switch (input_tensor_tmp.dtype()) {
                case diopi_dtype_bool: {
                    temp_bool = static_cast<bool>(*value);
                    value_ptr = &temp_bool;
                    break;
                }
                case diopi_dtype_int8: {
                    temp_i8 = int8_t(*value);
                    value_ptr = &temp_i8;
                    break;
                }
                case diopi_dtype_uint8: {
                    temp_u8 = uint8_t(*value);
                    value_ptr = &temp_u8;
                    break;
                }
                case diopi_dtype_int16: {
                    temp_i16 = int16_t(*value);
                    value_ptr = &temp_i16;
                    break;
                }
                case diopi_dtype_uint16: {
                    temp_u16 = uint16_t(*value);
                    value_ptr = &temp_u16;
                    break;
                }
                case diopi_dtype_int32: {
                    temp_i32 = int32_t(*value);
                    value_ptr = &temp_i32;
                    break;
                }
                case diopi_dtype_uint32: {
                    temp_u32 = uint32_t(*value);
                    value_ptr = &temp_u32;
                    break;
                }
                case diopi_dtype_int64: {
                    temp_i64 = int64_t(*value);
                    value_ptr = &temp_i64;
                    break;
                }
                case diopi_dtype_uint64: {
                    temp_u64 = uint64_t(*value);
                    value_ptr = &temp_u64;
                    break;
                }
                case diopi_dtype_float16: {
                    temp_f16 = half_float::half(*value);
                    value_ptr = &temp_f16;
                    break;
                }
                case diopi_dtype_float32: {
                    temp_f32 = static_cast<float>(*value);
                    value_ptr = &temp_f32;
                    break;
                }
            }
        }
        DIOPI_CALLCNNL(cnnlPad(
            handle, input_desc.get(), input_tensor_tmp.data(), new_pad, (value == nullptr) ? nullptr : value_ptr, out_desc.get(), out_tensor_tmp.data()));
    } else if (pad_mode == "reflect") {
        std::vector<int> input_dim = getDim(input_tensor_tmp);
        std::vector<int> out_dim = getDim(out_tensor_tmp);
        input_desc.set(input_tensor_tmp, CNNL_LAYOUT_NCHW, input_dim);
        out_desc.set(out_tensor_tmp, CNNL_LAYOUT_NCHW, out_dim);
        int pad_[4];
        if (pad_vec.size() == 4) {
            for (int i = 0; i < 4; i++) {
                pad_[i] = static_cast<int>(pad_vec[i]);
            }
        } else if (pad_vec.size() == 2) {
            pad_[2] = pad_[3] = 0;
            for (int i = 0; i < 2; i++) {
                pad_[i] = static_cast<int>(pad_vec[i]);
            }
        } else {
            DIOPI_CHECK(false, "Only supports 2D padding for reflection padding mode now.");
        }
        DIOPI_CALLCNNL(cnnlReflectionPad2d(handle, input_desc.get(), input_tensor_tmp.data(), pad_, out_desc.get(), out_tensor_tmp.data()));
    } else if (pad_mode == "replicate") {
        std::vector<int> input_dim = getDim(input_tensor_tmp);
        std::vector<int> out_dim = getDim(out_tensor_tmp);
        input_desc.set(input_tensor_tmp, CNNL_LAYOUT_NCHW, input_dim);
        out_desc.set(out_tensor_tmp, CNNL_LAYOUT_NCHW, out_dim);
        int pad_[4];
        if (pad_vec.size() == 4) {
            for (int i = 0; i < 4; i++) {
                pad_[i] = static_cast<int>(pad_vec[i]);
            }
        } else if (pad_vec.size() == 2) {
            pad_[2] = pad_[3] = 0;
            for (int i = 0; i < 2; i++) {
                pad_[i] = static_cast<int>(pad_vec[i]);
            }
        } else {
            DIOPI_CHECK(false, "Only supports 2D padding for replicate padding mode now.");
        }
        DIOPI_CALLCNNL(cnnlReplicationPad2d(handle, input_desc.get(), input_tensor_tmp.data(), pad_, out_desc.get(), out_tensor_tmp.data()));
    } else if (pad_mode == "circular") {
        input_desc.set(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
        out_desc.set(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

        auto create_slice_out = [&](auto& dst, auto src, int value, int dim) {
            std::vector<int64_t> slice_shape_1(src.shape().size());
            for (int i = 0; i < src.shape().size(); i++) {
                slice_shape_1[i] = src.shape()[i];
            }
            slice_shape_1[dim] = value;
            diopiSize_t slice_shape1(slice_shape_1.data(), slice_shape_1.size());
            DIOPI_CALL(diopiRequireTensor(ctx, &dst, &slice_shape1, nullptr, src.dtype(), diopi_device));
            return diopiSuccess;
        };

        auto slice_concat1 = [&](auto& dst, auto src, int start, int end, int dim) {
            diopiTensorHandle_t input_slice = nullptr;
            auto dim_value = end - start;
            DIOPI_CALL(create_slice_out(input_slice, src, dim_value, dim));
            DIOPI_CALL(diopiSlice(ctx, input_slice, static_cast<diopiTensorHandle_t>(src), dim, start, end, 1));
            diopiConstTensorHandle_t tensors_cat[2];
            tensors_cat[0] = static_cast<diopiConstTensorHandle_t>(src);
            tensors_cat[1] = static_cast<diopiConstTensorHandle_t>(input_slice);
            DIOPI_CALL(create_slice_out(dst, src, src.shape()[dim] + dim_value, dim));
            DIOPI_CALL(diopiCat(ctx, dst, tensors_cat, 2, dim));
            return diopiSuccess;
        };

        auto slice_concat2 = [&](auto& dst, auto src, int start, int end, int dim) {
            diopiTensorHandle_t input_slice = nullptr;
            auto dim_value = end - start;
            DIOPI_CALL(create_slice_out(input_slice, src, dim_value, dim));
            DIOPI_CALL(diopiSlice(ctx, input_slice, static_cast<diopiTensorHandle_t>(src), dim, start, end, 1));
            diopiConstTensorHandle_t tensors_cat[2];
            tensors_cat[0] = static_cast<diopiConstTensorHandle_t>(input_slice);
            tensors_cat[1] = static_cast<diopiConstTensorHandle_t>(src);
            DIOPI_CALL(create_slice_out(dst, src, src.shape()[dim] + dim_value, dim));
            DIOPI_CALL(diopiCat(ctx, dst, tensors_cat, 2, dim));
            return diopiSuccess;
        };

        diopiTensorHandle_t cat_out1 = nullptr;
        DIOPI_CALL(slice_concat1(cat_out1, input_tensor_tmp, 0, pad_vec[pad_vec.size() - 1], 2));

        DiopiTensor cat_out1_tensor(cat_out1);
        diopiTensorHandle_t cat_out2 = nullptr;
        DIOPI_CALL(slice_concat2(cat_out2, cat_out1_tensor, -(pad_vec[pad_vec.size() - 1] + pad_vec[pad_vec.size() - 2]), -pad_vec[pad_vec.size() - 1], 2));
        if (pad_vec.size() <= 2) {
            DIOPI_CALL(diopiCopyInp(ctx, cat_out2, static_cast<diopiTensorHandle_t>(out_tensor_tmp)));
        }

        if (pad_vec.size() > 2) {
            DiopiTensor cat_out2_tensor(cat_out2);
            diopiTensorHandle_t cat_out3 = nullptr;
            DIOPI_CALL(slice_concat1(cat_out3, cat_out2_tensor, 0, pad_vec[pad_vec.size() - 3], 3));

            DiopiTensor cat_out3_tensor(cat_out3);
            diopiTensorHandle_t cat_out4 = nullptr;
            DIOPI_CALL(slice_concat2(cat_out4, cat_out3_tensor, -(pad_vec[pad_vec.size() - 3] + pad_vec[pad_vec.size() - 4]), -pad_vec[pad_vec.size() - 3], 3));
            DIOPI_CALL(diopiCopyInp(ctx, cat_out4, static_cast<diopiTensorHandle_t>(out_tensor_tmp)));
        }
    } else {
        DIOPI_CHECK(false, "Only supports constant, reflect, circular and replicate now.");
    }
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
