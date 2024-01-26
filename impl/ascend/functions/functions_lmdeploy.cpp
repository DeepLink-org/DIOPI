/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>
#include <math.h>

#include <array>
#include <cassert>
#include <cstring>
#include <vector>

#define FLT_MIN __FLT_MIN__
#define FLT_MAX __FLT_MAX__

namespace impl {
namespace ascend {

// #define DIOPI_CHECK(expr)                                           \
//     do {                                                            \
//         diopiError_t ret = expr;                                    \
//         if (ret != diopiSuccess) {                                  \
//             printf(#expr " error at %s:%d.\n", __FILE__, __LINE__); \
//             return ret;                                             \
//         }                                                           \
//     } while (false);

// #define DIOPI_CHECK_FMT(expr, fmt, args...)                          \
//     do {                                                             \
//         diopiError_t ret = expr;                                     \
//         if (ret != diopiSuccess) {                                   \
//             printf(#fmt " at %s:%d.\n", ##args, __FILE__, __LINE__); \
//             return ret;                                              \
//         }                                                            \
//     } while (false);

int64_t* getDataOffsetPtr(diopiTensorHandle_t tensor, int64_t offset) {
    char* ptr = nullptr;
    diopiGetTensorData(tensor, reinterpret_cast<void**>(&ptr));
    if (offset == 0) {
        return reinterpret_cast<int64_t*>(ptr);
    }
    int64_t elem_size;
    diopiGetTensorElemSize(tensor, &elem_size);
    return reinterpret_cast<int64_t*>(ptr + offset * elem_size);
}

diopiError_t sliceAsSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    return diopiSlice(ctx, out, input, dim, index, index + 1, 1);
}

diopiError_t combAsCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    void* dataptr;
    diopiGetTensorData(out, &dataptr);
    int64_t itemsize;
    diopiGetTensorElemSize(out, &itemsize);
    diopiDevice_t device;
    diopiGetTensorDevice(out, &device);
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    diopiSize_t shapeinfo;
    diopiGetTensorShape(out, &shapeinfo);

    diopiTensorHandle_t forout;
    std::vector<int64_t> shape(shapeinfo.len);
    diopiSize_t newshape{shape.data(), shapeinfo.len};
    std::vector<int64_t> permuteshape(shapeinfo.len);
    diopiSize_t permutedims{permuteshape.data(), shapeinfo.len};
    int64_t offset = 1;
    for (int64_t i = 0; i < shapeinfo.len; i++) {
        if (i == dim) {
            shape[i] = shape[0];
            shape[0] = shapeinfo.data[i];
            permuteshape[0] = i;
            permuteshape[i] = 0;
        } else {
            shape[i] = shapeinfo.data[i];
            permuteshape[i] = i;
            offset *= shape[i];
        }
    }

    if (dim == 0) {
        int64_t totaldim0 = 0;
        for (int64_t i = 0; i < numInputs; i++) {
            diopiSize_t ishape;
            diopiGetTensorShape(tensors[i], &ishape);
            diopiDtype_t idtype;
            diopiGetTensorDtype(tensors[i], &idtype);
            if (idtype != dtype) {
                std::cout << "diff dtype in combAsCat" << std::endl;
                exit(-1);
            }
            diopiSize_t outi_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(reinterpret_cast<char*>(dataptr) + itemsize * totaldim0 * offset)),
                                    -1};
            diopiTensorHandle_t outi;
            shape[0] = ishape.data[0];
            diopiRequireTensor(ctx, &outi, &newshape, &outi_stride, dtype, device);
            diopiCopyInp(ctx, tensors[i], outi);
            totaldim0 += ishape.data[0];
        }
    } else {
        int64_t totaldimn = 0;
        diopiRequireTensor(ctx, &forout, &newshape, nullptr, dtype, device);
        for (int64_t i = 0; i < numInputs; i++) {
            diopiSize_t ishape;
            diopiGetTensorShape(tensors[i], &ishape);
            diopiDtype_t idtype;
            diopiGetTensorDtype(tensors[i], &idtype);
            if (idtype != dtype) {
                std::cout << "diff dtype in combAsCat" << std::endl;
                exit(-1);
            }
            diopiGetTensorData(forout, &dataptr);
            diopiSize_t outi_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(reinterpret_cast<char*>(dataptr) + itemsize * totaldimn * offset)),
                                    -1};
            diopiTensorHandle_t outi;
            shape[0] = ishape.data[dim];
            diopiRequireTensor(ctx, &outi, &newshape, &outi_stride, dtype, device);
            diopiPermute(ctx, outi, tensors[i], permutedims);
            totaldimn += ishape.data[dim];
        }
        diopiPermute(ctx, out, forout, permutedims);
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiFusedSiluFfnInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t weight1,
                                            diopiConstTensorHandle_t weight2, diopiConstTensorHandle_t weight3, diopiTensorHandle_t workspace,
                                            int64_t* workspace_size, int64_t fusion_level) {
    if (fusion_level >= 0) {
        diopiSize_t shapeinfo;
        diopiGetTensorShape(inoutput, &shapeinfo);
        int64_t token_num = shapeinfo.data[0];
        diopiGetTensorShape(weight1, &shapeinfo);
        int64_t inter_size = shapeinfo.data[1];
        int64_t itemsize = -1;
        diopiGetTensorElemSize(inoutput, &itemsize);
        if (*workspace_size < 0) {
            *workspace_size = 2 * itemsize * token_num * inter_size;
            return diopiSuccess;
        }
        void* dataptr;
        diopiGetTensorData(workspace, &dataptr);
        diopiDevice_t device;
        diopiGetTensorDevice(workspace, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(inoutput, &dtype);
        std::vector<int64_t> shape(2);
        diopiSize_t newshape{shape.data(), 2};
        shape[0] = token_num;
        shape[1] = inter_size;
        diopiSize_t strideW1{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(dataptr)), -1};
        diopiSize_t strideW3{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(reinterpret_cast<char*>(dataptr) + itemsize * token_num * inter_size)), -1};
        diopiTensorHandle_t matmulW1;
        diopiTensorHandle_t matmulW3;
        diopiRequireTensor(ctx, &matmulW1, &newshape, &strideW1, dtype, device);
        diopiRequireTensor(ctx, &matmulW3, &newshape, &strideW3, dtype, device);

        diopiMm(ctx, matmulW1, inoutput, weight1);
        diopiMm(ctx, matmulW3, inoutput, weight3);
        diopiSiluInp(ctx, matmulW1);
        diopiMulInp(ctx, matmulW1, matmulW3);
        diopiMm(ctx, inoutput, matmulW1, weight2);
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFusedContextAttentionInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t qkv_weight,
                                                     diopiConstTensorHandle_t qkv_bias, diopiTensorHandle_t pre_work, int64_t* pre_work_size, bool is_prepared,
                                                     diopiTensorHandle_t workspace, int64_t* workspace_size, int64_t fusion_level,
                                                     diopiTensorHandle_t* key_cache, diopiTensorHandle_t* value_cache, int64_t batch_size, diopiConstTensorHandle_t input_lengths,
                                                     diopiConstTensorHandle_t history_lengths, diopiConstTensorHandle_t context_lengths, int64_t layer_id,
                                                     int64_t local_head_num, int64_t local_kv_head_num, int64_t size_per_head, int64_t max_seq_len,
                                                     int64_t max_q_len, int64_t max_kv_len, int64_t rotary_embedding, float rope_theta) {
    if (fusion_level >= 0) {
        // workspace_size and pre_work_size
        diopiSize_t shapeinfo;
        diopiGetTensorShape(input_lengths, &shapeinfo);
        assert(batch_size == shapeinfo.data[0]);
        diopiDtype_t intdtype;
        diopiGetTensorDtype(input_lengths, &intdtype);
        int64_t intitemsize = -1;
        diopiGetTensorElemSize(input_lengths, &intitemsize);
        int64_t itemsize = -1;
        diopiGetTensorElemSize(inoutput, &itemsize);
        diopiSize_t inout_shapeinfo;
        diopiGetTensorShape(inoutput, &inout_shapeinfo);
        int64_t token_num = inout_shapeinfo.data[0];
        if (*workspace_size < 0 || *pre_work_size < 0) {
            *workspace_size = itemsize * token_num * (local_head_num + 2 * local_kv_head_num) * size_per_head +           // qkv_buffer
                              itemsize * batch_size * max_kv_len * local_kv_head_num * size_per_head +                    // k_cache_buffer
                              itemsize * batch_size * max_kv_len * local_kv_head_num * size_per_head +                    // v_cache_buffer
                              itemsize * batch_size * max_q_len * local_head_num * size_per_head +                        // q_cache_buffer
                              std::max(std::max(int64_t(sizeof(float) * max_q_len * local_head_num * size_per_head +      // timesteps and sphsteps buffer
                                                        itemsize * max_q_len * local_head_num * size_per_head +           // split
                                                        sizeof(float) * 2 * max_q_len * local_head_num * size_per_head),  // splitfp32 and catfp32
                                                int64_t(itemsize * max_seq_len * local_head_num * size_per_head * 5)),    // kv cal and cache with his
                                       int64_t(itemsize * batch_size * local_head_num * max_q_len * max_kv_len * 2)) +    // softmax
                              0;
            *pre_work_size = (itemsize * batch_size * max_q_len * max_kv_len + 31) / 32 * 32 +  // attention_mask_
                                                                                                // intitemsize * batch_size * max_q_len + // padding_offset_
                                                                                                // intitemsize * (batch_size + 1) + // cu_seqlens_
                             itemsize * max_seq_len * local_head_num * size_per_head +          // zeros
                             sizeof(float) * (max_seq_len + 32 + size_per_head / 2) +           // timesteps and sphsteps
                             0;
            return diopiSuccess;
        }

        void* workspace_ptr;
        diopiGetTensorData(workspace, &workspace_ptr);
        char* workspace1_ptr = reinterpret_cast<char*>(workspace_ptr) +                                           // workspace_ptr
                               itemsize * token_num * (local_head_num + 2 * local_kv_head_num) * size_per_head +  // qkv_buffer
                               itemsize * batch_size * max_kv_len * local_kv_head_num * size_per_head +           // k_cache_buffer
                               itemsize * batch_size * max_kv_len * local_kv_head_num * size_per_head +           // v_cache_buffer
                               itemsize * batch_size * max_q_len * local_head_num * size_per_head;                // q_cache_buffer
        void* prework_ptr;
        diopiGetTensorData(pre_work, &prework_ptr);
        diopiDevice_t device;
        diopiGetTensorDevice(inoutput, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(inoutput, &dtype);
        std::vector<int64_t> shape(4);
        diopiSize_t newshape{shape.data(), 4};
        // scalar zero
        diopiScalar_t scalar_dzero{dtype, double(0)};
        diopiScalar_t scalar_done{dtype, double(1)};
        // history_lengths_host and input_lengths_host and context_length_host
        shape[0] = batch_size;
        newshape.len = 1;
        diopiTensorHandle_t history_lengths_host;
        char history_lengths_host_data[batch_size * intitemsize];
        diopiSize_t history_lengths_host_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(history_lengths_host_data)), -1};
        diopiRequireTensor(ctx, &history_lengths_host, &newshape, &history_lengths_host_stride, intdtype, diopiDevice_t::diopi_host);
        diopiLmdeployCopyD2H(ctx, history_lengths_host, history_lengths, false);
        diopiTensorHandle_t input_lengths_host;
        char input_lengths_host_data[batch_size * intitemsize];
        diopiSize_t input_lengths_host_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(input_lengths_host_data)), -1};
        diopiRequireTensor(ctx, &input_lengths_host, &newshape, &input_lengths_host_stride, intdtype, diopiDevice_t::diopi_host);
        diopiLmdeployCopyD2H(ctx, input_lengths_host, input_lengths, false);
        diopiTensorHandle_t context_lengths_host;
        char context_lengths_host_data[batch_size * intitemsize];
        diopiSize_t context_lengths_host_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(context_lengths_host_data)), -1};
        diopiRequireTensor(ctx, &context_lengths_host, &newshape, &context_lengths_host_stride, intdtype, diopiDevice_t::diopi_host);
        diopiLmdeployCopyD2H(ctx, context_lengths_host, context_lengths, false);
        // attention_mask_ and padding_offset_ and cu_seqlens_
        diopiTensorHandle_t attention_mask_;
        newshape.len = 4;
        shape[0] = batch_size;
        shape[1] = 1;
        shape[2] = max_q_len;
        shape[3] = max_kv_len;
        diopiSize_t attention_mask_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(prework_ptr)), -1};
        diopiRequireTensor(ctx, &attention_mask_, &newshape, &attention_mask_stride, dtype, device);
        diopiTensorHandle_t zeros;
        newshape.len = 1;
        shape[0] = max_seq_len * local_head_num * size_per_head;
        char* zeros_ptr = reinterpret_cast<char*>(prework_ptr) + (itemsize * batch_size * max_q_len * max_kv_len + 31) / 32 * 32;
        diopiSize_t zeros_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(zeros_ptr)), -1};
        diopiRequireTensor(ctx, &zeros, &newshape, &zeros_stride, dtype, device);
        // ROPE prepare
        diopiDtype_t ropedtype = diopiDtype_t::diopi_dtype_float32;
        diopiTensorHandle_t timesteps;
        newshape.len = 1;
        shape[0] = max_seq_len + 32;
        char* timesteps_ptr = reinterpret_cast<char*>(zeros_ptr) + itemsize * max_seq_len * local_head_num * size_per_head;
        diopiSize_t timesteps_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timesteps_ptr)), -1};
        diopiRequireTensor(ctx, &timesteps, &newshape, &timesteps_stride, ropedtype, device);
        diopiTensorHandle_t sphsteps;
        newshape.len = 1;
        shape[0] = size_per_head / 2;
        char* sphsteps_ptr = reinterpret_cast<char*>(timesteps_ptr) + sizeof(float) * (max_seq_len + 32);
        diopiSize_t sphsteps_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_ptr)), -1};
        diopiRequireTensor(ctx, &sphsteps, &newshape, &sphsteps_stride, ropedtype, device);
        // prepared attention_mask_ and none padding_offset_ and none cu_seqlens_
        if (!is_prepared) {
            diopiScalar_t dnone{dtype, double(-1.0f)};
            diopiScalar_t d10000{dtype, double(10000.0f)};
            for (int64_t i = 0; i < batch_size; i++) {
                int64_t input_length = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(input_lengths_host_data) + i)
                                                                                   : *(reinterpret_cast<int64_t*>(input_lengths_host_data) + i);
                int64_t context_length = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(context_lengths_host_data) + i)
                                                                                     : *(reinterpret_cast<int64_t*>(context_lengths_host_data) + i);
                diopiConstTensorHandle_t attention_mask_members[3];
                int64_t attention_mask_members_length = 1;
                // attention_mask_i
                diopiTensorHandle_t attention_mask_i;
                shape[0] = max_q_len;
                shape[1] = max_kv_len;
                newshape.len = 2;
                char* attention_mask_i_ptr = reinterpret_cast<char*>(prework_ptr) + itemsize * i * max_q_len * max_kv_len;
                diopiSize_t attention_mask_i_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(attention_mask_i_ptr)), -1};
                diopiRequireTensor(ctx, &attention_mask_i, &newshape, &attention_mask_i_stride, dtype, device);
                diopiTensorHandle_t mask_i_upper_part;
                shape[0] = input_length;
                shape[1] = max_kv_len;
                newshape.len = 2;
                char* mask_i_upper_part_ptr = reinterpret_cast<char*>(attention_mask_i_ptr);
                diopiSize_t mask_i_upper_part_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(mask_i_upper_part_ptr)), -1};
                diopiRequireTensor(ctx, &mask_i_upper_part, &newshape, &mask_i_upper_part_stride, dtype, device);
                diopiTensorHandle_t mask_i_lower_part;
                shape[0] = max_q_len - input_length;
                shape[1] = max_kv_len;
                newshape.len = 2;
                char* mask_i_lower_part_ptr = reinterpret_cast<char*>(mask_i_upper_part_ptr) + itemsize * input_length * max_kv_len;
                diopiSize_t mask_i_lower_part_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(mask_i_lower_part_ptr)), -1};
                diopiRequireTensor(ctx, &mask_i_lower_part, &newshape, &mask_i_lower_part_stride, dtype, device);

                // attention_mask_mask
                diopiTensorHandle_t attention_mask_mask;
                shape[0] = input_length;
                shape[1] = input_length;
                newshape.len = 2;
                char* attention_mask_mask_ptr = reinterpret_cast<char*>(workspace_ptr);
                diopiSize_t attention_mask_mask_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(attention_mask_mask_ptr)), -1};
                diopiRequireTensor(ctx, &attention_mask_mask, &newshape, &attention_mask_mask_stride, dtype, device);
                diopiFill(ctx, attention_mask_mask, &dnone);
                diopiTriuInp(ctx, attention_mask_mask, 1);
                // attention_mask_zero
                int64_t context_input = std::max(int64_t(0), int64_t(context_length - input_length));
                if (context_input > 0) {
                    diopiTensorHandle_t attention_mask_zero;
                    shape[0] = input_length;
                    shape[1] = context_input;
                    newshape.len = 2;
                    char* attention_mask_zero_ptr = reinterpret_cast<char*>(attention_mask_mask_ptr) + itemsize * input_length * input_length;
                    diopiSize_t attention_mask_zero_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(attention_mask_zero_ptr)), -1};
                    diopiRequireTensor(ctx, &attention_mask_zero, &newshape, &attention_mask_zero_stride, dtype, device);
                    diopiFill(ctx, attention_mask_zero, &scalar_dzero);
                    attention_mask_members[0] = attention_mask_zero;
                    attention_mask_members[1] = attention_mask_mask;
                    attention_mask_members_length += 1;
                } else {
                    attention_mask_members[0] = attention_mask_mask;
                }
                // attention_mask_one
                int64_t others = std::max(int64_t(0), int64_t(max_kv_len - context_length));
                if (others > 0) {
                    diopiTensorHandle_t attention_mask_one;
                    shape[0] = input_length;
                    shape[1] = others;
                    newshape.len = 2;
                    char* attention_mask_one_ptr = reinterpret_cast<char*>(attention_mask_mask_ptr) + itemsize * input_length * (context_input + input_length);
                    diopiSize_t attention_mask_one_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(attention_mask_one_ptr)), -1};
                    diopiRequireTensor(ctx, &attention_mask_one, &newshape, &attention_mask_one_stride, dtype, device);
                    diopiFill(ctx, attention_mask_one, &dnone);
                    attention_mask_members[attention_mask_members_length] = attention_mask_one;
                    attention_mask_members_length += 1;
                }
                // cat
                combAsCat(ctx, mask_i_upper_part, attention_mask_members, attention_mask_members_length, 1);
                diopiFill(ctx, mask_i_lower_part, &dnone);
                
                // for cal
                diopiMulInpScalar(ctx, attention_mask_i, &d10000);
            }
            // zeros
            diopiFill(ctx, zeros, &scalar_dzero);
            // ROPE prepare
            diopiTensorHandle_t sphsteps_buff;
            newshape.len = 1;
            shape[0] = size_per_head / 2;
            char* sphsteps_buff_ptr = reinterpret_cast<char*>(workspace_ptr);
            diopiSize_t sphsteps_buff_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_buff_ptr)), -1};
            diopiRequireTensor(ctx, &sphsteps_buff, &newshape, &sphsteps_buff_stride, ropedtype, device);
            diopiScalar_t sphsteps_start{ropedtype, double(0)};
            diopiScalar_t sphsteps_end{ropedtype, double((size_per_head / 2 - 1) * 2)};  // == size_per_head -2 and size_per_head always be even
            diopiLinspace(ctx, sphsteps_buff, &sphsteps_start, &sphsteps_end, size_per_head / 2);
            diopiScalar_t theta{ropedtype, double(rope_theta)};
            diopiScalar_t embedding{ropedtype, double(rotary_embedding)};
            diopiDivInpScalar(ctx, sphsteps_buff, &embedding, diopiRoundMode_t::RoundModeNone);
            diopiPowScalar(ctx, sphsteps, &theta, sphsteps_buff);

            diopiScalar_t rope_start{ropedtype, double(0)};
            diopiScalar_t rope_end{ropedtype, double(max_seq_len + 32 - 1)};
            diopiLinspace(ctx, timesteps, &rope_start, &rope_end, max_seq_len + 32);
            return diopiSuccess;
        }
        // cal qkv
        diopiTensorHandle_t qkv_buffer;
        shape[0] = token_num;
        shape[1] = (local_head_num + 2 * local_kv_head_num) * size_per_head;
        newshape.len = 2;
        char* qkv_buffer_ptr = reinterpret_cast<char*>(workspace_ptr);
        diopiSize_t qkv_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qkv_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &qkv_buffer, &newshape, &qkv_buffer_stride, dtype, device);
        diopiMm(ctx, qkv_buffer, inoutput, qkv_weight);
        // split q,k,v
        diopiTensorHandle_t q_buffer;
        shape[0] = token_num;
        shape[1] = local_head_num;
        shape[2] = size_per_head;
        newshape.len = 3;
        char* q_buffer_ptr = reinterpret_cast<char*>(qkv_buffer_ptr);
        diopiSize_t q_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &q_buffer, &newshape, &q_buffer_stride, dtype, device);
        diopiTensorHandle_t k_buffer;
        shape[1] = local_kv_head_num;
        char* k_buffer_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * token_num * local_head_num * size_per_head;
        diopiSize_t k_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &k_buffer, &newshape, &k_buffer_stride, dtype, device);
        diopiTensorHandle_t v_buffer;
        shape[1] = local_kv_head_num;
        char* v_buffer_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * token_num * (local_head_num + local_kv_head_num) * size_per_head;
        diopiSize_t v_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &v_buffer, &newshape, &v_buffer_stride, dtype, device);
        // split q,k,v temp
        diopiTensorHandle_t qkv_buffer_forsplit;
        shape[0] = token_num;
        shape[1] = (local_head_num + 2 * local_kv_head_num);
        shape[2] = size_per_head;
        newshape.len = 3;
        diopiRequireTensor(ctx, &qkv_buffer_forsplit, &newshape, &qkv_buffer_stride, dtype, device);
        diopiTensorHandle_t q_buffer_temp;
        shape[0] = token_num;
        shape[1] = local_head_num;
        shape[2] = size_per_head;
        newshape.len = 3;
        char* q_buffer_temp_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * token_num * (local_head_num + local_kv_head_num * 2) * size_per_head;
        diopiSize_t q_buffer_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_temp_ptr)), -1};
        diopiRequireTensor(ctx, &q_buffer_temp, &newshape, &q_buffer_temp_stride, dtype, device);
        diopiTensorHandle_t k_buffer_temp;
        shape[1] = local_kv_head_num;
        char* k_buffer_temp_ptr = reinterpret_cast<char*>(q_buffer_temp_ptr) + itemsize * token_num * local_head_num * size_per_head;
        diopiSize_t k_buffer_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_buffer_temp_ptr)), -1};
        diopiRequireTensor(ctx, &k_buffer_temp, &newshape, &k_buffer_temp_stride, dtype, device);
        diopiTensorHandle_t v_buffer_temp;
        shape[1] = local_kv_head_num;
        char* v_buffer_temp_ptr = reinterpret_cast<char*>(k_buffer_temp_ptr) + itemsize * token_num * local_kv_head_num * size_per_head;
        diopiSize_t v_buffer_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_buffer_temp_ptr)), -1};
        diopiRequireTensor(ctx, &v_buffer_temp, &newshape, &v_buffer_temp_stride, dtype, device);
        diopiTensorHandle_t qkv_buffers[3]{q_buffer_temp, k_buffer_temp, v_buffer_temp};
        std::vector<int64_t> qkv_buffers_split_sizes_data{local_head_num, local_kv_head_num, local_kv_head_num};
        diopiSize_t qkv_buffers_split_sizes{qkv_buffers_split_sizes_data.data(), 3};
        diopiSplitWithSizes(ctx, qkv_buffers, 3, qkv_buffer_forsplit, qkv_buffers_split_sizes, 1);
        // q,k,v bias
        if (qkv_bias != nullptr) {
            const void* qkv_bias_ptr;
            diopiGetTensorDataConst(qkv_bias, &qkv_bias_ptr);
            diopiTensorHandle_t q_bias;
            diopiTensorHandle_t k_bias;
            diopiTensorHandle_t v_bias;
            shape[0] = 1;
            shape[1] = local_head_num;
            shape[2] = size_per_head;
            newshape.len = 3;
            char* q_bias_ptr = reinterpret_cast<char*>(const_cast<void*>(qkv_bias_ptr));
            diopiSize_t q_bias_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_bias_ptr)), -1};
            diopiRequireTensor(ctx, &q_bias, &newshape, &q_bias_stride, dtype, device);
            shape[1] = local_kv_head_num;
            char* k_bias_ptr = q_bias_ptr + itemsize * local_head_num * size_per_head;
            diopiSize_t k_bias_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_bias_ptr)), -1};
            diopiRequireTensor(ctx, &k_bias, &newshape, &k_bias_stride, dtype, device);
            char* v_bias_ptr = k_bias_ptr + itemsize * local_kv_head_num * size_per_head;
            diopiSize_t v_bias_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_bias_ptr)), -1};
            diopiRequireTensor(ctx, &v_bias, &newshape, &v_bias_stride, dtype, device);
            diopiAdd(ctx, q_buffer, q_buffer_temp, q_bias, &scalar_done);
            diopiAdd(ctx, k_buffer, k_buffer_temp, k_bias, &scalar_done);
            diopiAdd(ctx, v_buffer, v_buffer_temp, v_bias, &scalar_done);
        } else {
            diopiLmdeployCopyD2D(ctx, q_buffer, q_buffer_temp, false);
            diopiLmdeployCopyD2D(ctx, k_buffer, k_buffer_temp, false);
            diopiLmdeployCopyD2D(ctx, v_buffer, v_buffer_temp, false);
        }
        // copy to kv cache and get kv
        // k_cache_buf_ and v_cache_buf_ for attn and q_cache_buf_ for transposed q_buffer
        diopiTensorHandle_t k_cache_buf_;
        shape[0] = batch_size * local_head_num;
        shape[1] = size_per_head;
        shape[2] = max_kv_len;
        newshape.len = 3;
        char* k_cache_buf_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * token_num * (local_head_num + 2 * local_kv_head_num) * size_per_head;
        diopiSize_t k_cache_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_cache_buf_ptr)), -1};
        diopiRequireTensor(ctx, &k_cache_buf_, &newshape, &k_cache_buf_stride, dtype, device);
        diopiTensorHandle_t v_cache_buf_;
        shape[0] = batch_size * local_head_num;
        shape[1] = max_kv_len;
        shape[2] = size_per_head;
        char* v_cache_buf_ptr = reinterpret_cast<char*>(k_cache_buf_ptr) + itemsize * batch_size * local_kv_head_num * size_per_head * max_kv_len;
        diopiSize_t v_cache_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_cache_buf_ptr)), -1};
        diopiRequireTensor(ctx, &v_cache_buf_, &newshape, &v_cache_buf_stride, dtype, device);
        diopiTensorHandle_t q_cache_buf_;
        shape[1] = max_q_len;
        char* q_cache_buf_ptr = reinterpret_cast<char*>(v_cache_buf_ptr) + itemsize * batch_size * local_kv_head_num * size_per_head * max_kv_len;
        diopiSize_t q_cache_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_cache_buf_ptr)), -1};
        diopiRequireTensor(ctx, &q_cache_buf_, &newshape, &q_cache_buf_stride, dtype, device);
        // k v cache
        int64_t layer_offset = layer_id * local_kv_head_num * max_seq_len * size_per_head * itemsize;
        // transpose info
        std::vector<int64_t> trans102_data{1, 0, 2};
        diopiSize_t trans102{trans102_data.data(), 3};
        std::vector<int64_t> trans120_data{1, 2, 0};
        diopiSize_t trans120{trans120_data.data(), 3};
        if (local_head_num == local_kv_head_num) {
            int64_t total_input_length = 0;
            for (int64_t i = 0; i < batch_size; i++) {
                int64_t input_length = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(input_lengths_host_data) + i)
                                                                                   : *(reinterpret_cast<int64_t*>(input_lengths_host_data) + i);
                int64_t history_length = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(history_lengths_host_data) + i)
                                                                                     : *(reinterpret_cast<int64_t*>(history_lengths_host_data) + i);
                // ROPE begin
                diopiScalar_t rope_start{ropedtype, double(history_length)};
                diopiScalar_t rope_end{ropedtype, double(history_length + input_length - 1)};
                diopiTensorHandle_t timestep;
                newshape.len = 1;
                shape[0] = input_length;
                char* timestep_ptr = reinterpret_cast<char*>(timesteps_ptr) + sizeof(float) * history_length;
                diopiSize_t timestep_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timestep_ptr)), -1};
                diopiRequireTensor(ctx, &timestep, &newshape, &timestep_ptr_stride, ropedtype, device);
                diopiTensorHandle_t timestep_buff_32;
                newshape.len = 4;
                shape[0] = input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head / 2;
                shape[3] = 1;
                char* timestep_buff_32_ptr = reinterpret_cast<char*>(workspace1_ptr);
                diopiSize_t timestep_buff_32_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timestep_buff_32_ptr)), -1};
                diopiRequireTensor(ctx, &timestep_buff_32, &newshape, &timestep_buff_32_stride, ropedtype, device);
                diopiTensorHandle_t timestep_forexpand;
                newshape.len = 4;
                shape[0] = input_length;
                shape[1] = 1;
                shape[2] = 1;
                shape[3] = 1;
                diopiRequireTensor(ctx, &timestep_forexpand, &newshape, &timestep_ptr_stride, ropedtype, device);
                diopiExpand(ctx, timestep_buff_32, timestep_forexpand);
                diopiTensorHandle_t sphstep_buff_32;
                newshape.len = 4;
                shape[0] = input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head / 2;
                shape[3] = 1;
                char* sphstep_buff_32_ptr = reinterpret_cast<char*>(timestep_buff_32_ptr) + sizeof(float) * input_length * local_head_num * size_per_head / 2;
                diopiSize_t sphstep_buff_32_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphstep_buff_32_ptr)), -1};
                diopiRequireTensor(ctx, &sphstep_buff_32, &newshape, &sphstep_buff_32_stride, ropedtype, device);
                diopiTensorHandle_t sphstep;
                newshape.len = 4;
                shape[0] = 1;
                shape[1] = 1;
                shape[2] = size_per_head / 2;
                shape[3] = 1;
                diopiSize_t sphstep_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_ptr)), -1};
                diopiRequireTensor(ctx, &sphstep, &newshape, &sphstep_ptr_stride, ropedtype, device);
                diopiExpand(ctx, sphstep_buff_32, sphstep);

                // return diopiSuccess; // SH
                diopiDivInp(ctx, timestep_buff_32, sphstep_buff_32, diopiRoundMode_t::RoundModeNone);
                diopiSin(ctx, sphstep_buff_32, timestep_buff_32);  // sphsteps_buff as sin
                diopiCosInp(ctx, timestep_buff_32);                // timesteps_buff as cos

                // diopiTensorHandle_t timestep_buff;
                // newshape.len = 4;
                // shape[0] = input_length;
                // shape[1] = local_head_num;
                // shape[2] = size_per_head / 2;
                // shape[3] = 1;
                // diopiRequireTensor(ctx, &timestep_buff, &newshape, nullptr, dtype, device);
                // diopiTensorHandle_t sphstep_buff;
                // newshape.len = 4;
                // shape[0] = input_length;
                // shape[1] = local_head_num;
                // shape[2] = size_per_head / 2;
                // shape[3] = 1;
                // diopiRequireTensor(ctx, &sphstep_buff, &newshape, nullptr, dtype, device);
                // diopiCastDtype(ctx, timestep_buff, timestep_buff_32);
                // diopiCastDtype(ctx, sphstep_buff, sphstep_buff_32);

                diopiTensorHandle_t split0_buffer;  // x0
                diopiTensorHandle_t split1_buffer;  // x1
                newshape.len = 4;
                shape[0] = input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head / 2;
                shape[3] = 1;
                char* split0_buffer_ptr = reinterpret_cast<char*>(sphstep_buff_32_ptr) + sizeof(float) * input_length * local_head_num * size_per_head / 2;
                diopiSize_t split0_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(split0_buffer_ptr)), -1};
                diopiRequireTensor(ctx, &split0_buffer, &newshape, &split0_buffer_stride, dtype, device);
                char* split1_buffer_ptr = reinterpret_cast<char*>(split0_buffer_ptr) + itemsize * input_length * local_head_num * size_per_head / 2;
                diopiSize_t split1_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(split1_buffer_ptr)), -1};
                diopiRequireTensor(ctx, &split1_buffer, &newshape, &split1_buffer_stride, dtype, device);
                diopiConstTensorHandle_t const_splits_buffer[2] = {split0_buffer, split1_buffer};

                diopiTensorHandle_t split0_buffer_32;  // x0
                diopiTensorHandle_t split1_buffer_32;  // x1
                char* split0_buffer_32_ptr = reinterpret_cast<char*>(split1_buffer_ptr) + itemsize * input_length * local_head_num * size_per_head / 2;
                diopiSize_t split0_buffer_32_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(split0_buffer_32_ptr)), -1};
                diopiRequireTensor(ctx, &split0_buffer_32, &newshape, &split0_buffer_32_stride, ropedtype, device);
                char* split1_buffer_32_ptr = reinterpret_cast<char*>(split0_buffer_32_ptr) + sizeof(float) * input_length * local_head_num * size_per_head / 2;
                diopiSize_t split1_buffer_32_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(split1_buffer_32_ptr)), -1};
                diopiRequireTensor(ctx, &split1_buffer_32, &newshape, &split1_buffer_32_stride, ropedtype, device);

                diopiTensorHandle_t cat0_buffer;  // x0
                diopiTensorHandle_t cat1_buffer;  // x1
                char* cat0_buffer_ptr = reinterpret_cast<char*>(split1_buffer_32_ptr) + sizeof(float) * input_length * local_head_num * size_per_head / 2;
                diopiSize_t cat0_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(cat0_buffer_ptr)), -1};
                diopiRequireTensor(ctx, &cat0_buffer, &newshape, &cat0_buffer_stride, ropedtype, device);
                char* cat1_buffer_ptr = reinterpret_cast<char*>(cat0_buffer_ptr) + sizeof(float) * input_length * local_head_num * size_per_head / 2;
                diopiSize_t cat1_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(cat1_buffer_ptr)), -1};
                diopiRequireTensor(ctx, &cat1_buffer, &newshape, &cat1_buffer_stride, ropedtype, device);
                diopiTensorHandle_t splits_buffer[2] = {split0_buffer, split1_buffer};
                diopiConstTensorHandle_t cat_buffer[2] = {cat0_buffer, cat1_buffer};
                std::vector<int64_t> split_sizes_data{1, 1};
                diopiSize_t split_sizes{split_sizes_data.data(), 2};
                // q
                diopiTensorHandle_t q_buffer_forsplit;
                newshape.len = 4;
                shape[0] = input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head / 2;
                shape[3] = 2;
                char* q_buffer_forsplit_ptr = reinterpret_cast<char*>(q_buffer_ptr) + itemsize * total_input_length * local_head_num * size_per_head;
                diopiSize_t q_buffer_forsplit_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_forsplit_ptr)), -1};
                diopiRequireTensor(ctx, &q_buffer_forsplit, &newshape, &q_buffer_forsplit_stride, dtype, device);
                diopiSplitWithSizes(ctx, splits_buffer, 2, q_buffer_forsplit, split_sizes, 3);

                diopiCastDtype(ctx, split0_buffer_32, split0_buffer);
                diopiCastDtype(ctx, split1_buffer_32, split1_buffer);
                // 0
                diopiMul(ctx, cat0_buffer, split0_buffer_32, timestep_buff_32);
                diopiMul(ctx, cat1_buffer, split1_buffer_32, sphstep_buff_32);
                diopiSubInp(ctx, cat0_buffer, cat1_buffer, &scalar_done);
                // 1
                diopiMulInp(ctx, split0_buffer_32, sphstep_buff_32);
                diopiMulInp(ctx, split1_buffer_32, timestep_buff_32);
                diopiAdd(ctx, cat1_buffer, split0_buffer_32, split1_buffer_32, &scalar_done);

                diopiCastDtype(ctx, split0_buffer, cat0_buffer);
                diopiCastDtype(ctx, split1_buffer, cat1_buffer);
                combAsCat(ctx, q_buffer_forsplit, const_splits_buffer, 2, 3);
                // cat
                // combAsCat(ctx, q_buffer_forsplit, cat_buffer, 2, 3);
                // k
                diopiTensorHandle_t k_buffer_forsplit;
                newshape.len = 4;
                shape[0] = input_length;
                shape[1] = local_kv_head_num;
                shape[2] = size_per_head / 2;
                shape[3] = 2;
                char* k_buffer_forsplit_ptr = reinterpret_cast<char*>(k_buffer_ptr) + itemsize * total_input_length * local_kv_head_num * size_per_head;
                diopiSize_t k_buffer_forsplit_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_buffer_forsplit_ptr)), -1};
                diopiRequireTensor(ctx, &k_buffer_forsplit, &newshape, &k_buffer_forsplit_stride, dtype, device);
                diopiSplitWithSizes(ctx, splits_buffer, 2, k_buffer_forsplit, split_sizes, 3);

                diopiCastDtype(ctx, split0_buffer_32, split0_buffer);
                diopiCastDtype(ctx, split1_buffer_32, split1_buffer);
                // 0
                diopiMul(ctx, cat0_buffer, split0_buffer_32, timestep_buff_32);
                diopiMul(ctx, cat1_buffer, split1_buffer_32, sphstep_buff_32);
                diopiSubInp(ctx, cat0_buffer, cat1_buffer, &scalar_done);
                // 1
                diopiMulInp(ctx, split0_buffer_32, sphstep_buff_32);
                diopiMulInp(ctx, split1_buffer_32, timestep_buff_32);
                diopiAdd(ctx, cat1_buffer, split0_buffer_32, split1_buffer_32, &scalar_done);

                diopiCastDtype(ctx, split0_buffer, cat0_buffer);
                diopiCastDtype(ctx, split1_buffer, cat1_buffer);
                combAsCat(ctx, k_buffer_forsplit, const_splits_buffer, 2, 3);
                // cat
                // combAsCat(ctx, k_buffer_forsplit, cat_buffer, 2, 3);
                // cache begin
                // q
                diopiTensorHandle_t q_cal;  // q buffer with cache for attn
                newshape.len = 3;
                shape[0] = local_head_num;
                shape[1] = max_q_len;
                shape[2] = size_per_head;
                char* q_cal_ptr = reinterpret_cast<char*>(q_cache_buf_ptr) + itemsize * i * local_head_num * max_q_len * size_per_head;
                diopiSize_t q_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_cal_ptr)), -1};
                diopiRequireTensor(ctx, &q_cal, &newshape, &q_cal_stride, dtype, device);
                diopiTensorHandle_t prepared_q_buffer;
                newshape.len = 3;
                shape[0] = input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head;
                diopiRequireTensor(ctx, &prepared_q_buffer, &newshape, &q_buffer_forsplit_stride, dtype, device);
                diopiTensorHandle_t zeros_q;
                newshape.len = 3;
                shape[0] = max_q_len - input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head;
                diopiSize_t zeros_q_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(zeros_ptr)), -1};
                diopiRequireTensor(ctx, &zeros_q, &newshape, &zeros_q_stride, dtype, device);
                diopiTensorHandle_t qcal102;
                newshape.len = 3;
                shape[0] = max_q_len;
                shape[1] = local_head_num;
                shape[2] = size_per_head;
                char* qcal102_ptr = reinterpret_cast<char*>(workspace1_ptr);
                diopiSize_t qcal102_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qcal102_ptr)), -1};
                diopiRequireTensor(ctx, &qcal102, &newshape, &qcal102_stride, dtype, device);
                diopiConstTensorHandle_t cat2qcal[2] = {prepared_q_buffer, zeros_q};
                combAsCat(ctx, qcal102, cat2qcal, 2, 0);
                diopiPermute(ctx, q_cal, qcal102, trans102);
                // k
                diopiTensorHandle_t k_cal;  // k buffer with cache for attn
                newshape.len = 3;
                shape[0] = local_kv_head_num;
                shape[1] = size_per_head;
                shape[2] = max_kv_len;
                char* k_cal_ptr = reinterpret_cast<char*>(k_cache_buf_ptr) + itemsize * i * local_kv_head_num * max_kv_len * size_per_head;
                diopiSize_t k_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_cal_ptr)), -1};
                diopiRequireTensor(ctx, &k_cal, &newshape, &k_cal_stride, dtype, device);
                diopiTensorHandle_t prepared_k_buffer;
                newshape.len = 3;
                shape[0] = input_length;
                shape[1] = local_kv_head_num;
                shape[2] = size_per_head;
                diopiRequireTensor(ctx, &prepared_k_buffer, &newshape, &k_buffer_forsplit_stride, dtype, device);
                // v
                diopiTensorHandle_t v_cal;  // v buffer with cache for attn
                newshape.len = 3;
                shape[0] = local_kv_head_num;
                shape[1] = max_kv_len;
                shape[2] = size_per_head;
                char* v_cal_ptr = reinterpret_cast<char*>(v_cache_buf_ptr) + itemsize * i * local_kv_head_num * max_kv_len * size_per_head;
                diopiSize_t v_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_cal_ptr)), -1};
                diopiRequireTensor(ctx, &v_cal, &newshape, &v_cal_stride, dtype, device);
                diopiTensorHandle_t prepared_v_buffer;
                newshape.len = 3;
                shape[0] = input_length;
                shape[1] = local_kv_head_num;
                shape[2] = size_per_head;
                diopiSize_t v_buffer_forsplit_stride{
                    static_cast<const int64_t*>(
                        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(v_buffer_ptr) + itemsize * total_input_length * local_kv_head_num * size_per_head)),
                    -1};
                diopiRequireTensor(ctx, &prepared_v_buffer, &newshape, &v_buffer_forsplit_stride, dtype, device);
                // k cache
                diopiTensorHandle_t k_cache;
                newshape.len = 3;
                shape[0] = local_kv_head_num;
                shape[1] = max_seq_len;
                shape[2] = size_per_head;
                void* key_cache_ptr;
                diopiGetTensorData(key_cache[i], &key_cache_ptr);
                char* k_cache_ptr = reinterpret_cast<char*>(key_cache_ptr) + layer_offset;
                diopiSize_t k_cache_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_cache_ptr)), -1};
                diopiRequireTensor(ctx, &k_cache, &newshape, &k_cache_stride, dtype, device);
                // v cache
                diopiTensorHandle_t v_cache;
                newshape.len = 3;
                shape[0] = local_kv_head_num;
                shape[1] = max_seq_len;
                shape[2] = size_per_head;
                void* value_cache_ptr;
                diopiGetTensorData(value_cache[i], &value_cache_ptr);
                char* v_cache_ptr = reinterpret_cast<char*>(value_cache_ptr) + layer_offset;
                diopiSize_t v_cache_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_cache_ptr)), -1};
                diopiRequireTensor(ctx, &v_cache, &newshape, &v_cache_stride, dtype, device);
                if (history_length == 0) {
                    // transpose info
                    if (max_kv_len > input_length) {
                        // k for cal
                        diopiTensorHandle_t zeros_kv;
                        newshape.len = 3;
                        shape[0] = max_kv_len - input_length;
                        shape[1] = local_kv_head_num;
                        shape[2] = size_per_head;
                        diopiSize_t zeros_kv_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(zeros_ptr)), -1};
                        diopiRequireTensor(ctx, &zeros_kv, &newshape, &zeros_kv_stride, dtype, device);
                        diopiTensorHandle_t kvcal;
                        newshape.len = 3;
                        shape[0] = max_kv_len;
                        shape[1] = local_kv_head_num;
                        shape[2] = size_per_head;
                        char* kvcal_ptr = reinterpret_cast<char*>(workspace1_ptr);
                        diopiSize_t kvcal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(kvcal_ptr)), -1};
                        diopiRequireTensor(ctx, &kvcal, &newshape, &kvcal_stride, dtype, device);
                        diopiConstTensorHandle_t cat2kcal[2] = {prepared_k_buffer, zeros_kv};
                        combAsCat(ctx, kvcal, cat2kcal, 2, 0);
                        diopiPermute(ctx, k_cal, kvcal, trans120);
                        // v for cal
                        diopiConstTensorHandle_t cat2vcal[2] = {prepared_v_buffer, zeros_kv};
                        combAsCat(ctx, kvcal, cat2vcal, 2, 0);
                        diopiPermute(ctx, v_cal, kvcal, trans102);
                    } else {
                        diopiPermute(ctx, k_cal, prepared_k_buffer, trans120);
                        diopiPermute(ctx, v_cal, prepared_v_buffer, trans102);
                    }

                    if (max_seq_len > input_length) {
                        diopiTensorHandle_t zeros_kv_all;
                        newshape.len = 3;
                        shape[0] = max_seq_len - input_length;
                        shape[1] = local_kv_head_num;
                        shape[2] = size_per_head;
                        diopiSize_t zeros_kv_all_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(zeros_ptr)), -1};
                        diopiRequireTensor(ctx, &zeros_kv_all, &newshape, &zeros_kv_all_stride, dtype, device);
                        diopiTensorHandle_t kvcache102;
                        newshape.len = 3;
                        shape[0] = max_seq_len;
                        shape[1] = local_kv_head_num;
                        shape[2] = size_per_head;
                        char* kvcache102_ptr = reinterpret_cast<char*>(workspace1_ptr);
                        diopiSize_t kvcache102_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(kvcache102_ptr)), -1};
                        diopiRequireTensor(ctx, &kvcache102, &newshape, &kvcache102_stride, dtype, device);
                        diopiConstTensorHandle_t cat2kcache[2] = {prepared_k_buffer, zeros_kv_all};
                        combAsCat(ctx, kvcache102, cat2kcache, 2, 0);
                        diopiPermute(ctx, k_cache, kvcache102, trans102);
                        diopiConstTensorHandle_t cat2vcache[2] = {prepared_v_buffer, zeros_kv_all};
                        combAsCat(ctx, kvcache102, cat2vcache, 2, 0);
                        diopiPermute(ctx, v_cache, kvcache102, trans102);
                    } else {
                        diopiPermute(ctx, k_cache, prepared_k_buffer, trans102);
                        diopiPermute(ctx, v_cache, prepared_v_buffer, trans102);
                    }
                } else {
                    // transpose info
                    diopiTensorHandle_t his_k;
                    newshape.len = 3;
                    shape[0] = local_kv_head_num;
                    shape[1] = history_length;
                    shape[2] = size_per_head;
                    char* his_k_ptr = reinterpret_cast<char*>(workspace1_ptr);
                    diopiSize_t his_k_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(his_k_ptr)), -1};
                    diopiRequireTensor(ctx, &his_k, &newshape, &his_k_stride, dtype, device);
                    diopiSlice(ctx, his_k, k_cache, 1, 0, history_length, 1);
                    diopiTensorHandle_t his_v;
                    char* his_v_ptr = reinterpret_cast<char*>(his_k_ptr) + itemsize * local_kv_head_num * history_length * size_per_head;
                    diopiSize_t his_v_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(his_v_ptr)), -1};
                    diopiRequireTensor(ctx, &his_v, &newshape, &his_v_stride, dtype, device);
                    diopiSlice(ctx, his_v, v_cache, 1, 0, history_length, 1);
                    std::vector<int64_t> trans120_data{1, 2, 0};
                    diopiSize_t trans120{trans120_data.data(), 3};
                    diopiTensorHandle_t his_k102;
                    newshape.len = 3;
                    shape[0] = history_length;
                    shape[1] = local_kv_head_num;
                    shape[2] = size_per_head;
                    char* his_k102_ptr = reinterpret_cast<char*>(his_v_ptr) + itemsize * local_kv_head_num * history_length * size_per_head;
                    diopiSize_t his_k102_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(his_k102_ptr)), -1};
                    diopiRequireTensor(ctx, &his_k102, &newshape, &his_k102_stride, dtype, device);
                    diopiTensorHandle_t his_v102;
                    char* his_v102_ptr = reinterpret_cast<char*>(his_k102_ptr) + itemsize * local_kv_head_num * history_length * size_per_head;
                    diopiSize_t his_v102_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(his_v102_ptr)), -1};
                    diopiRequireTensor(ctx, &his_v102, &newshape, &his_v102_stride, dtype, device);
                    diopiPermute(ctx, his_k102, his_k, trans102);
                    diopiPermute(ctx, his_v102, his_v, trans102);

                    diopiTensorHandle_t kvcal;
                    newshape.len = 3;
                    shape[0] = max_kv_len;
                    shape[1] = local_kv_head_num;
                    shape[2] = size_per_head;
                    char* kvcal_ptr = reinterpret_cast<char*>(his_v102_ptr) + itemsize * local_kv_head_num * history_length * size_per_head;
                    diopiSize_t kvcal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(kvcal_ptr)), -1};
                    diopiRequireTensor(ctx, &kvcal, &newshape, &kvcal_stride, dtype, device);
                    if (max_kv_len > input_length + history_length) {
                        // k for cal
                        diopiTensorHandle_t zeros_kv;
                        newshape.len = 3;
                        shape[0] = max_kv_len - input_length - history_length;
                        shape[1] = local_kv_head_num;
                        shape[2] = size_per_head;
                        diopiSize_t zeros_kv_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(zeros_ptr)), -1};
                        diopiRequireTensor(ctx, &zeros_kv, &newshape, &zeros_kv_stride, dtype, device);
                        diopiConstTensorHandle_t cat2kcal[3] = {his_k102, prepared_k_buffer, zeros_kv};
                        combAsCat(ctx, kvcal, cat2kcal, 3, 0);
                        diopiPermute(ctx, k_cal, kvcal, trans120);
                        // v for cal
                        diopiConstTensorHandle_t cat2vcal[3] = {his_v102, prepared_v_buffer, zeros_kv};
                        combAsCat(ctx, kvcal, cat2vcal, 3, 0);
                        diopiPermute(ctx, v_cal, kvcal, trans102);
                    } else {
                        diopiConstTensorHandle_t cat2kcal[2] = {his_k102, prepared_k_buffer};
                        combAsCat(ctx, kvcal, cat2kcal, 2, 0);
                        diopiPermute(ctx, k_cal, kvcal, trans120);
                        // v for cal
                        diopiConstTensorHandle_t cat2vcal[2] = {his_v102, prepared_v_buffer};
                        combAsCat(ctx, kvcal, cat2vcal, 2, 0);
                        diopiPermute(ctx, v_cal, kvcal, trans102);
                    }

                    diopiTensorHandle_t kvcache102;
                    newshape.len = 3;
                    shape[0] = max_seq_len;
                    shape[1] = local_kv_head_num;
                    shape[2] = size_per_head;
                    char* kvcache102_ptr = reinterpret_cast<char*>(his_v102_ptr) + itemsize * local_kv_head_num * history_length * size_per_head;
                    diopiSize_t kvcache102_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(kvcache102_ptr)), -1};
                    diopiRequireTensor(ctx, &kvcache102, &newshape, &kvcache102_stride, dtype, device);
                    if (max_seq_len > input_length + history_length) {
                        diopiTensorHandle_t zeros_kv_all;
                        newshape.len = 3;
                        shape[0] = max_seq_len - input_length - history_length;
                        shape[1] = local_kv_head_num;
                        shape[2] = size_per_head;
                        diopiSize_t zeros_kv_all_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(zeros_ptr)), -1};
                        diopiRequireTensor(ctx, &zeros_kv_all, &newshape, &zeros_kv_all_stride, dtype, device);
                        diopiConstTensorHandle_t cat2kcache[3] = {his_k102, prepared_k_buffer, zeros_kv_all};
                        combAsCat(ctx, kvcache102, cat2kcache, 3, 0);
                        diopiPermute(ctx, k_cache, kvcache102, trans102);
                        diopiConstTensorHandle_t cat2vcache[3] = {his_v102, prepared_v_buffer, zeros_kv_all};
                        combAsCat(ctx, kvcache102, cat2vcache, 3, 0);
                        diopiPermute(ctx, v_cache, kvcache102, trans102);
                    } else {
                        diopiConstTensorHandle_t cat2kcache[2] = {his_k102, prepared_k_buffer};
                        combAsCat(ctx, kvcache102, cat2kcache, 2, 0);
                        diopiPermute(ctx, k_cache, kvcache102, trans102);
                        diopiConstTensorHandle_t cat2vcache[2] = {his_v102, prepared_v_buffer};
                        combAsCat(ctx, kvcache102, cat2vcache, 2, 0);
                        diopiPermute(ctx, v_cache, kvcache102, trans102);
                    }
                }
                total_input_length += input_length;
            }
            // attn
            diopiTensorHandle_t qk_buffer;
            newshape.len = 3;
            shape[0] = batch_size * local_head_num;
            shape[1] = max_q_len;
            shape[2] = max_kv_len;
            char* qk_buffer_ptr = reinterpret_cast<char*>(workspace1_ptr);
            diopiSize_t qk_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qk_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &qk_buffer, &newshape, &qk_buffer_stride, dtype, device);
            diopiBmm(ctx, qk_buffer, q_cache_buf_, k_cache_buf_);
            diopiScalar_t qk_scale{dtype, double(1.f / sqrtf(size_per_head * 1.f))};
            diopiMulInpScalar(ctx, qk_buffer, &qk_scale);

            diopiTensorHandle_t qk_buffer_formask;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = max_q_len;
            shape[3] = max_kv_len;
            diopiSize_t qk_buffer_formask_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qk_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &qk_buffer_formask, &newshape, &qk_buffer_formask_stride, dtype, device);
            diopiAddInp(ctx, qk_buffer_formask, attention_mask_, &scalar_done);
            // diopiSoftmax() with eps and -max
            diopiTensorHandle_t qk_softmax;
            newshape.len = 3;
            shape[0] = batch_size * local_head_num;
            shape[1] = max_q_len;
            shape[2] = max_kv_len;
            char* qk_softmax_ptr = reinterpret_cast<char*>(qk_buffer_ptr) + itemsize * batch_size * local_head_num * max_q_len * max_kv_len;
            diopiSize_t qk_softmax_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qk_softmax_ptr)), -1};
            diopiRequireTensor(ctx, &qk_softmax, &newshape, &qk_softmax_stride, dtype, device);
            diopiSoftmax(ctx, qk_softmax, qk_buffer, 2);
            // * V
            diopiBmm(ctx, q_cache_buf_, qk_softmax, v_cache_buf_);

            // to out
            total_input_length = 0;
            for (int64_t i = 0; i < batch_size; i++) {
                int64_t input_length = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(input_lengths_host_data) + i)
                                                                                   : *(reinterpret_cast<int64_t*>(input_lengths_host_data) + i);
                int64_t history_length = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(history_lengths_host_data) + i)
                                                                                     : *(reinterpret_cast<int64_t*>(history_lengths_host_data) + i);
                diopiTensorHandle_t q_withpad;
                newshape.len = 3;
                shape[0] = local_head_num;
                shape[1] = max_q_len;
                shape[2] = size_per_head;
                char* q_withpad_ptr = reinterpret_cast<char*>(q_cache_buf_ptr) + itemsize * i * local_head_num * max_q_len * size_per_head;
                diopiSize_t q_withpad_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_withpad_ptr)), -1};
                diopiRequireTensor(ctx, &q_withpad, &newshape, &q_withpad_stride, dtype, device);
                diopiTensorHandle_t q_withoutpad;
                newshape.len = 3;
                shape[0] = local_head_num;
                shape[1] = input_length;
                shape[2] = size_per_head;
                char* q_withoutpad_ptr = reinterpret_cast<char*>(workspace1_ptr);
                diopiSize_t q_withoutpad_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_withoutpad_ptr)), -1};
                diopiRequireTensor(ctx, &q_withoutpad, &newshape, &q_withoutpad_stride, dtype, device);
                diopiSlice(ctx, q_withoutpad, q_withpad, 1, 0, input_length, 1);
                diopiTensorHandle_t q_out;
                newshape.len = 3;
                shape[0] = input_length;
                shape[1] = local_head_num;
                shape[2] = size_per_head;
                void* inout_ptr;
                diopiGetTensorData(inoutput, &inout_ptr);
                char* q_out_ptr = reinterpret_cast<char*>(inout_ptr) + itemsize * total_input_length * local_head_num * size_per_head;
                diopiSize_t q_out_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_out_ptr)), -1};
                diopiRequireTensor(ctx, &q_out, &newshape, &q_out_ptr_stride, dtype, device);
                diopiPermute(ctx, q_out, q_withoutpad, trans102);
                total_input_length += input_length;
            }
        } else {
            return diopiErrorOccurred;
        }

        return diopiSuccess;
    }
    return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiFusedDecoderAttentionInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t qkv_weight,
                                                     diopiConstTensorHandle_t qkv_bias, diopiTensorHandle_t workspace, int64_t* workspace_size,
                                                     int64_t fusion_level, diopiTensorHandle_t* key_cache, diopiTensorHandle_t* value_cache,
                                                     diopiConstTensorHandle_t finished, diopiConstTensorHandle_t total_padding_tokens,
                                                     diopiConstTensorHandle_t sequence_lengths, int64_t step, int64_t layer_id, int64_t local_head_num,
                                                     int64_t local_kv_head_num, int64_t size_per_head, int64_t max_seq_len, int64_t rotary_embedding,
                                                     float rope_theta) {
    if (fusion_level >= 0) {
        // workspace_size
        diopiSize_t shapeinfo;
        diopiGetTensorShape(inoutput, &shapeinfo);
        int64_t batch_size = shapeinfo.data[0];
        int64_t intitemsize = -1;
        diopiGetTensorElemSize(sequence_lengths, &intitemsize);
        int64_t itemsize = -1;
        diopiGetTensorElemSize(inoutput, &itemsize);
        if (*workspace_size < 0) {
            *workspace_size = itemsize * batch_size * (local_head_num + 2 * local_kv_head_num) * size_per_head +      // qkv_buffer
                              itemsize * batch_size * local_head_num * size_per_head * max_seq_len +                  // k_cahce_buffer
                              itemsize * batch_size * local_head_num * size_per_head * max_seq_len +                  // v_cahce_buffer
                              itemsize * batch_size * local_head_num * size_per_head * max_seq_len +                  // q_cahce_buffer
                              std::max(int64_t(sizeof(float) * (batch_size + size_per_head) +                         // timesteps + sphsteps + sphsteps_temp
                                               sizeof(float) * batch_size * local_head_num * size_per_head +          // timesteps_buff_32 sphsteps_buff_32
                                               itemsize * batch_size * local_head_num * size_per_head +               // timesteps_buff sphsteps_buff
                                               itemsize * batch_size * local_head_num * size_per_head * 2),           // split and cat
                                       int64_t(itemsize * local_head_num * (max_seq_len + 1) * size_per_head * 3)) +  // or ki cal
                              0;
            return diopiSuccess;
        }
        bool same = false;

        std::vector<int64_t> shape(4);
        diopiSize_t newshape{shape.data(), 4};
        void* workspace_ptr;
        diopiGetTensorData(workspace, &workspace_ptr);
        char* workspace1_ptr = reinterpret_cast<char*>(workspace_ptr) +
                               itemsize * batch_size * (local_head_num + 2 * local_kv_head_num) * size_per_head +  // qkv_buffer
                               itemsize * batch_size * local_head_num * size_per_head * max_seq_len +              // k_cahce_buffer
                               itemsize * batch_size * local_head_num * size_per_head * max_seq_len +              // v_cahce_buffer
                               itemsize * batch_size * local_head_num * size_per_head * max_seq_len;               // q_cahce_buffer
        diopiDtype_t intdtype;
        diopiGetTensorDtype(sequence_lengths, &intdtype);
        void* inout_ptr;
        diopiGetTensorData(inoutput, &inout_ptr);
        diopiDevice_t device;
        diopiGetTensorDevice(inoutput, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(inoutput, &dtype);

        // scalar one
        diopiScalar_t scalar_done{dtype, double(1)};

        diopiTensorHandle_t qkv_buffer;
        shape[0] = batch_size;
        shape[1] = (local_head_num + 2 * local_kv_head_num) * size_per_head;
        newshape.len = 2;
        char* qkv_buffer_ptr = reinterpret_cast<char*>(workspace_ptr);
        diopiSize_t qkv_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qkv_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &qkv_buffer, &newshape, &qkv_buffer_stride, dtype, device);
        diopiMm(ctx, qkv_buffer, inoutput, qkv_weight);
        // split q,k,v
        diopiTensorHandle_t q_buffer;
        shape[0] = batch_size;
        shape[1] = local_head_num;
        shape[2] = size_per_head;
        newshape.len = 3;
        char* q_buffer_ptr = reinterpret_cast<char*>(qkv_buffer_ptr);
        diopiSize_t q_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &q_buffer, &newshape, &q_buffer_stride, dtype, device);
        diopiTensorHandle_t k_buffer;
        shape[1] = local_kv_head_num;
        char* k_buffer_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * batch_size * local_head_num * size_per_head;
        diopiSize_t k_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &k_buffer, &newshape, &k_buffer_stride, dtype, device);
        diopiTensorHandle_t v_buffer;
        shape[1] = local_kv_head_num;
        char* v_buffer_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * batch_size * (local_head_num + local_kv_head_num) * size_per_head;
        diopiSize_t v_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &v_buffer, &newshape, &v_buffer_stride, dtype, device);

        // split q,k,v temp
        diopiTensorHandle_t qkv_buffer_forsplit;
        shape[0] = batch_size;
        shape[1] = (local_head_num + 2 * local_kv_head_num);
        shape[2] = size_per_head;
        newshape.len = 3;
        diopiRequireTensor(ctx, &qkv_buffer_forsplit, &newshape, &qkv_buffer_stride, dtype, device);
        diopiTensorHandle_t q_buffer_temp;
        shape[0] = batch_size;
        shape[1] = local_head_num;
        shape[2] = size_per_head;
        newshape.len = 3;
        char* q_buffer_temp_ptr = reinterpret_cast<char*>(qkv_buffer_ptr) + itemsize * batch_size * (local_head_num + local_kv_head_num * 2) * size_per_head;
        diopiSize_t q_buffer_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_temp_ptr)), -1};
        diopiRequireTensor(ctx, &q_buffer_temp, &newshape, &q_buffer_temp_stride, dtype, device);
        diopiTensorHandle_t k_buffer_temp;
        shape[1] = local_kv_head_num;
        char* k_buffer_temp_ptr = reinterpret_cast<char*>(q_buffer_temp_ptr) + itemsize * batch_size * local_head_num * size_per_head;
        diopiSize_t k_buffer_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_buffer_temp_ptr)), -1};
        diopiRequireTensor(ctx, &k_buffer_temp, &newshape, &k_buffer_temp_stride, dtype, device);
        diopiTensorHandle_t v_buffer_temp;
        shape[1] = local_kv_head_num;
        char* v_buffer_temp_ptr = reinterpret_cast<char*>(k_buffer_temp_ptr) + itemsize * batch_size * local_kv_head_num * size_per_head;
        diopiSize_t v_buffer_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_buffer_temp_ptr)), -1};
        diopiRequireTensor(ctx, &v_buffer_temp, &newshape, &v_buffer_temp_stride, dtype, device);
        diopiTensorHandle_t qkv_buffers[3]{q_buffer_temp, k_buffer_temp, v_buffer_temp};
        std::vector<int64_t> qkv_buffers_split_sizes_data{local_head_num, local_kv_head_num, local_kv_head_num};
        diopiSize_t qkv_buffers_split_sizes{qkv_buffers_split_sizes_data.data(), 3};
        diopiSplitWithSizes(ctx, qkv_buffers, 3, qkv_buffer_forsplit, qkv_buffers_split_sizes, 1);
        // q,k,v bias
        if (qkv_bias != nullptr) {
            const void* qkv_bias_ptr;
            diopiGetTensorDataConst(qkv_bias, &qkv_bias_ptr);
            diopiTensorHandle_t q_bias;
            diopiTensorHandle_t k_bias;
            diopiTensorHandle_t v_bias;
            shape[0] = 1;
            shape[1] = local_head_num;
            shape[2] = size_per_head;
            newshape.len = 3;
            char* q_bias_ptr = reinterpret_cast<char*>(const_cast<void*>(qkv_bias_ptr));
            diopiSize_t q_bias_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_bias_ptr)), -1};
            diopiRequireTensor(ctx, &q_bias, &newshape, &q_bias_stride, dtype, device);
            shape[1] = local_kv_head_num;
            char* k_bias_ptr = q_bias_ptr + itemsize * local_head_num * size_per_head;
            diopiSize_t k_bias_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_bias_ptr)), -1};
            diopiRequireTensor(ctx, &k_bias, &newshape, &k_bias_stride, dtype, device);
            char* v_bias_ptr = k_bias_ptr + itemsize * local_kv_head_num * size_per_head;
            diopiSize_t v_bias_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_bias_ptr)), -1};
            diopiRequireTensor(ctx, &v_bias, &newshape, &v_bias_stride, dtype, device);
            diopiAdd(ctx, q_buffer, q_buffer_temp, q_bias, &scalar_done);
            diopiAdd(ctx, k_buffer, k_buffer_temp, k_bias, &scalar_done);
            diopiAdd(ctx, v_buffer, v_buffer_temp, v_bias, &scalar_done);
        } else {
            diopiLmdeployCopyD2D(ctx, q_buffer, q_buffer_temp, false);
            diopiLmdeployCopyD2D(ctx, k_buffer, k_buffer_temp, false);
            diopiLmdeployCopyD2D(ctx, v_buffer, v_buffer_temp, false);
        }
        // finished
        shape[0] = batch_size;
        newshape.len = 1;
        diopiTensorHandle_t finished_host;
        bool finished_host_data[batch_size];
        diopiSize_t finished_host_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(finished_host_data)), -1};
        diopiRequireTensor(ctx, &finished_host, &newshape, &finished_host_stride, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
        diopiLmdeployCopyD2H(ctx, finished_host, finished, false);
        // keep same
        diopiTensorHandle_t q_buffer_back;
        if (same) {
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head;
            newshape.len = 3;
            diopiRequireTensor(ctx, &q_buffer_back, &newshape, nullptr, dtype, device);
            for (int64_t i = 0; i < batch_size; i++) {
                if (finished_host_data[i]) {
                    diopiTensorHandle_t q_buffer_back_one;
                    shape[0] = local_head_num;
                    shape[1] = size_per_head;
                    newshape.len = 2;
                    void* q_buffer_back_one_ptr;
                    diopiGetTensorData(q_buffer_back, &q_buffer_back_one_ptr);
                    diopiSize_t q_buffer_back_one_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(reinterpret_cast<char*>(q_buffer_back_one_ptr) +
                                                                                                                i * itemsize * local_head_num * size_per_head)),
                                                         -1};
                    diopiRequireTensor(ctx, &q_buffer_back_one, &newshape, &q_buffer_back_one_stride, dtype, device);
                    diopiTensorHandle_t q_buffer_one;
                    shape[0] = local_head_num;
                    shape[1] = size_per_head;
                    newshape.len = 2;
                    char* q_buffer_one_ptr = q_buffer_ptr + i * itemsize * local_head_num * size_per_head;
                    diopiSize_t q_buffer_one_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_one_ptr)), -1};
                    diopiRequireTensor(ctx, &q_buffer_one, &newshape, &q_buffer_one_stride, dtype, device);
                    diopiLmdeployCopyD2D(ctx, q_buffer_back_one, q_buffer_one, true);
                }
            }
        }

        shape[0] = batch_size;
        newshape.len = 1;
        diopiTensorHandle_t sequence_lengths_host;
        char sequence_lengths_host_data[batch_size * intitemsize];
        diopiSize_t sequence_lengths_host_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sequence_lengths_host_data)), -1};
        diopiRequireTensor(ctx, &sequence_lengths_host, &newshape, &sequence_lengths_host_stride, intdtype, diopiDevice_t::diopi_host);
        diopiLmdeployCopyD2H(ctx, sequence_lengths_host, sequence_lengths, false);
        // scalar_dstep
        diopiScalar_t scalar_dstep{dtype, double(step - 1)};
        // ROPE
        if (rotary_embedding > 0) {
            // ROPE prepare
            diopiDtype_t ropedtype = diopiDtype_t::diopi_dtype_float32;
            diopiTensorHandle_t timesteps;
            newshape.len = 1;
            shape[0] = batch_size;
            char* timesteps_ptr = reinterpret_cast<char*>(workspace1_ptr);
            diopiSize_t timesteps_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timesteps_ptr)), -1};
            diopiRequireTensor(ctx, &timesteps, &newshape, &timesteps_stride, ropedtype, device);
            diopiFill(ctx, timesteps, &scalar_dstep);
            diopiSubInp(ctx, timesteps, total_padding_tokens, &scalar_done);
            diopiTensorHandle_t sphsteps;
            newshape.len = 1;
            shape[0] = size_per_head / 2;
            char* sphsteps_ptr = reinterpret_cast<char*>(workspace1_ptr) + sizeof(float) * batch_size;
            diopiSize_t sphsteps_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_ptr)), -1};
            diopiRequireTensor(ctx, &sphsteps, &newshape, &sphsteps_stride, ropedtype, device);
            diopiTensorHandle_t sphsteps_temp;
            char* sphsteps_temp_ptr = reinterpret_cast<char*>(sphsteps_ptr) + sizeof(float) * size_per_head / 2;
            diopiSize_t sphsteps_temp_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_temp_ptr)), -1};
            diopiRequireTensor(ctx, &sphsteps_temp, &newshape, &sphsteps_temp_stride, ropedtype, device);
            diopiScalar_t sphsteps_start{ropedtype, double(0)};
            diopiScalar_t sphsteps_end{ropedtype, double((size_per_head / 2 - 1) * 2)};  // == size_per_head -2 and size_per_head always be even
            diopiLinspace(ctx, sphsteps_temp, &sphsteps_start, &sphsteps_end, size_per_head / 2);
            diopiScalar_t theta{ropedtype, double(rope_theta)};
            diopiScalar_t embedding{ropedtype, double(rotary_embedding)};
            diopiDivInpScalar(ctx, sphsteps_temp, &embedding, diopiRoundMode_t::RoundModeNone);
            diopiPowScalar(ctx, sphsteps, &theta, sphsteps_temp);
            // ROPE begin
            diopiTensorHandle_t timesteps_buff_32;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head / 2;
            shape[3] = 1;
            char* timesteps_buff_32_ptr = reinterpret_cast<char*>(sphsteps_temp_ptr) + sizeof(float) * size_per_head / 2;
            diopiSize_t timesteps_buff_32_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timesteps_buff_32_ptr)), -1};
            diopiRequireTensor(ctx, &timesteps_buff_32, &newshape, &timesteps_buff_32_stride, ropedtype, device);
            diopiTensorHandle_t timesteps_forexpand;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = 1;
            shape[2] = 1;
            shape[3] = 1;
            diopiSize_t timesteps_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timesteps_ptr)), -1};
            diopiRequireTensor(ctx, &timesteps_forexpand, &newshape, &timesteps_ptr_stride, ropedtype, device);
            diopiExpand(ctx, timesteps_buff_32, timesteps_forexpand);
            diopiTensorHandle_t sphsteps_buff_32;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head / 2;
            shape[3] = 1;
            char* sphsteps_buff_32_ptr = reinterpret_cast<char*>(timesteps_buff_32_ptr) + sizeof(float) * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t sphsteps_buff_32_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_buff_32_ptr)), -1};
            diopiRequireTensor(ctx, &sphsteps_buff_32, &newshape, &sphsteps_buff_32_stride, ropedtype, device);
            diopiTensorHandle_t sphsteps_forexpand;
            newshape.len = 4;
            shape[0] = 1;
            shape[1] = 1;
            shape[2] = size_per_head / 2;
            shape[3] = 1;
            diopiSize_t sphsteps_ptr_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_ptr)), -1};
            diopiRequireTensor(ctx, &sphsteps_forexpand, &newshape, &sphsteps_ptr_stride, ropedtype, device);
            diopiExpand(ctx, sphsteps_buff_32, sphsteps_forexpand);
            diopiDivInp(ctx, timesteps_buff_32, sphsteps_buff_32, diopiRoundMode_t::RoundModeNone);
            diopiSin(ctx, sphsteps_buff_32, timesteps_buff_32);  // sphsteps_buff as sin
            diopiCosInp(ctx, timesteps_buff_32);                 // timesteps_buff as cos

            diopiTensorHandle_t timesteps_buff;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head / 2;
            shape[3] = 1;
            char* timesteps_buff_ptr = reinterpret_cast<char*>(sphsteps_buff_32_ptr) + sizeof(float) * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t timesteps_buff_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(timesteps_buff_ptr)), -1};
            diopiRequireTensor(ctx, &timesteps_buff, &newshape, &timesteps_buff_stride, dtype, device);
            diopiTensorHandle_t sphsteps_buff;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head / 2;
            shape[3] = 1;
            char* sphsteps_buff_ptr = reinterpret_cast<char*>(timesteps_buff_ptr) + itemsize * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t sphsteps_buff_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(sphsteps_buff_ptr)), -1};
            diopiRequireTensor(ctx, &sphsteps_buff, &newshape, &sphsteps_buff_stride, dtype, device);
            diopiCastDtype(ctx, timesteps_buff, timesteps_buff_32);
            diopiCastDtype(ctx, sphsteps_buff, sphsteps_buff_32);

            diopiTensorHandle_t split0_buffer;  // x0
            diopiTensorHandle_t split1_buffer;  // x1
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head / 2;
            shape[3] = 1;
            char* split0_buffer_ptr = reinterpret_cast<char*>(sphsteps_buff_ptr) + itemsize * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t split0_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(split0_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &split0_buffer, &newshape, &split0_buffer_stride, dtype, device);
            char* split1_buffer_ptr = reinterpret_cast<char*>(split0_buffer_ptr) + itemsize * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t split1_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(split1_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &split1_buffer, &newshape, &split1_buffer_stride, dtype, device);
            diopiTensorHandle_t cat0_buffer;  // x0
            diopiTensorHandle_t cat1_buffer;  // x1
            char* cat0_buffer_ptr = reinterpret_cast<char*>(split1_buffer_ptr) + itemsize * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t cat0_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(cat0_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &cat0_buffer, &newshape, &cat0_buffer_stride, dtype, device);
            char* cat1_buffer_ptr = reinterpret_cast<char*>(cat0_buffer_ptr) + itemsize * batch_size * local_head_num * size_per_head / 2;
            diopiSize_t cat1_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(cat1_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &cat1_buffer, &newshape, &cat1_buffer_stride, dtype, device);
            diopiTensorHandle_t splits_buffer[2] = {split0_buffer, split1_buffer};
            diopiConstTensorHandle_t cat_buffer[2] = {cat0_buffer, cat1_buffer};
            std::vector<int64_t> split_sizes_data{1, 1};
            diopiSize_t split_sizes{split_sizes_data.data(), 2};
            // q
            diopiTensorHandle_t q_buffer_forsplit;
            newshape.len = 4;
            shape[0] = batch_size;
            shape[1] = local_head_num;
            shape[2] = size_per_head / 2;
            shape[3] = 2;
            diopiSize_t q_buffer_forsplit_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_ptr)), -1};
            diopiRequireTensor(ctx, &q_buffer_forsplit, &newshape, &q_buffer_forsplit_stride, dtype, device);
            diopiSplitWithSizes(ctx, splits_buffer, 2, q_buffer_forsplit, split_sizes, 3);
            // 0
            diopiMul(ctx, cat0_buffer, split0_buffer, timesteps_buff);
            diopiMul(ctx, cat1_buffer, split1_buffer, sphsteps_buff);
            diopiSubInp(ctx, cat0_buffer, cat1_buffer, &scalar_done);
            // 1
            diopiMulInp(ctx, split0_buffer, sphsteps_buff);
            diopiMulInp(ctx, split1_buffer, timesteps_buff);
            diopiAdd(ctx, cat1_buffer, split0_buffer, split1_buffer, &scalar_done);
            // cat
            combAsCat(ctx, q_buffer_forsplit, cat_buffer, 2, 3);
            if (local_kv_head_num == local_head_num) {
                // k
                diopiTensorHandle_t k_buffer_forsplit;
                newshape.len = 4;
                shape[0] = batch_size;
                shape[1] = local_kv_head_num;
                shape[2] = size_per_head / 2;
                shape[3] = 2;
                diopiSize_t k_buffer_forsplit_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_buffer_ptr)), -1};
                diopiRequireTensor(ctx, &k_buffer_forsplit, &newshape, &k_buffer_forsplit_stride, dtype, device);
                diopiSplitWithSizes(ctx, splits_buffer, 2, k_buffer_forsplit, split_sizes, 3);
                // 0
                diopiMul(ctx, cat0_buffer, split0_buffer, timesteps_buff);
                diopiMul(ctx, cat1_buffer, split1_buffer, sphsteps_buff);
                diopiSubInp(ctx, cat0_buffer, cat1_buffer, &scalar_done);
                // 1
                diopiMulInp(ctx, split0_buffer, sphsteps_buff);
                diopiMulInp(ctx, split1_buffer, timesteps_buff);
                diopiAdd(ctx, cat1_buffer, split0_buffer, split1_buffer, &scalar_done);
                // cat
                combAsCat(ctx, k_buffer_forsplit, cat_buffer, 2, 3);
            } else {
                return diopiErrorOccurred;
            }
            // kv cache offset
            int64_t kv_cache_layer_offset = layer_id * local_kv_head_num * max_seq_len * size_per_head * itemsize;
            // transpose info
            std::vector<int64_t> trans021_data{0, 2, 1};
            diopiSize_t trans021{trans021_data.data(), 3};
            // qk scale
            diopiScalar_t inv_sqrt_dh{dtype, double(1.F / (sqrtf(float(size_per_head) * 1.f)))};
            for (int64_t i = 0; i < batch_size; i++) {
                if (local_head_num == local_kv_head_num) {
                    if (!finished_host_data[i]) {
                        int64_t sequence_lengths = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(sequence_lengths_host_data) + i)
                                                                                               : *(reinterpret_cast<int64_t*>(sequence_lengths_host_data) + i);
                        const int64_t tlength = sequence_lengths;
                        const int64_t first_step = std::max(int64_t(0), int64_t(tlength + 1 - max_seq_len));
                        const int64_t tlength_circ = tlength % max_seq_len;
                        // qi for cal
                        diopiTensorHandle_t qi_cal;
                        newshape.len = 3;
                        shape[0] = local_head_num;
                        shape[1] = 1;
                        shape[2] = size_per_head;
                        char* qi_cal_ptr = reinterpret_cast<char*>(q_buffer_ptr) + i * itemsize * local_head_num * size_per_head;
                        diopiSize_t qi_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qi_cal_ptr)), -1};
                        diopiRequireTensor(ctx, &qi_cal, &newshape, &qi_cal_stride, dtype, device);
                        // ki_buffer
                        diopiTensorHandle_t ki_buffer1;
                        newshape.len = 3;
                        shape[0] = local_kv_head_num;
                        shape[1] = 1;
                        shape[2] = size_per_head;
                        char* ki_buffer_ptr = reinterpret_cast<char*>(k_buffer_ptr) + i * itemsize * local_kv_head_num * size_per_head;
                        diopiSize_t ki_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(ki_buffer_ptr)), -1};
                        diopiRequireTensor(ctx, &ki_buffer1, &newshape, &ki_buffer_stride, dtype, device);
                        diopiTensorHandle_t ki_buffer2;
                        shape[0] = local_kv_head_num;
                        shape[1] = size_per_head;
                        shape[2] = 1;
                        diopiRequireTensor(ctx, &ki_buffer2, &newshape, &ki_buffer_stride, dtype, device);
                        // k cache
                        diopiTensorHandle_t k_cache;
                        newshape.len = 3;
                        shape[0] = local_kv_head_num;
                        shape[1] = max_seq_len;
                        shape[2] = size_per_head;
                        void* key_cache_ptr;
                        diopiGetTensorData(key_cache[i], &key_cache_ptr);
                        char* k_cache_ptr = reinterpret_cast<char*>(key_cache_ptr) + kv_cache_layer_offset;
                        diopiSize_t k_cache_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(k_cache_ptr)), -1};
                        diopiRequireTensor(ctx, &k_cache, &newshape, &k_cache_stride, dtype, device);
                        // update k cache, index is tlength_circ
                        diopiConstTensorHandle_t cat_kcache[3];
                        int64_t cat_kcache_length = 1;
                        if (tlength_circ > 0) {
                            diopiTensorHandle_t ki_cache_beg;
                            shape[0] = local_kv_head_num;
                            shape[1] = tlength_circ;
                            shape[2] = size_per_head;
                            char* ki_cache_beg_ptr = reinterpret_cast<char*>(workspace1_ptr);
                            diopiSize_t ki_cache_beg_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(ki_cache_beg_ptr)), -1};
                            diopiRequireTensor(ctx, &ki_cache_beg, &newshape, &ki_cache_beg_stride, dtype, device);
                            diopiSlice(ctx, ki_cache_beg, k_cache, 1, 0, tlength_circ, 1);
                            cat_kcache[0] = ki_cache_beg;
                            cat_kcache[1] = ki_buffer1;
                            cat_kcache_length += 1;
                        } else {
                            cat_kcache[0] = ki_buffer1;
                        }
                        if (tlength_circ != max_seq_len - 1) {
                            diopiTensorHandle_t ki_cache_end;
                            shape[0] = local_kv_head_num;
                            shape[1] = max_seq_len - tlength_circ - 1;
                            shape[2] = size_per_head;
                            char* ki_cache_end_ptr = reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_kv_head_num * tlength_circ * size_per_head;
                            diopiSize_t ki_cache_end_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(ki_cache_end_ptr)), -1};
                            diopiRequireTensor(ctx, &ki_cache_end, &newshape, &ki_cache_end_stride, dtype, device);
                            diopiSlice(ctx, ki_cache_end, k_cache, 1, tlength_circ + 1, max_seq_len, 1);
                            cat_kcache[cat_kcache_length] = ki_cache_end;
                            cat_kcache_length += 1;
                        }
                        combAsCat(ctx, k_cache, cat_kcache, cat_kcache_length, 1);
                        // ki for cal
                        diopiTensorHandle_t ki_cal;
                        newshape.len = 3;
                        shape[0] = local_head_num;
                        shape[1] = size_per_head;
                        shape[2] = tlength - first_step + 1;
                        char* ki_cal_ptr = reinterpret_cast<char*>(workspace1_ptr);
                        diopiSize_t ki_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(ki_cal_ptr)), -1};
                        diopiRequireTensor(ctx, &ki_cal, &newshape, &ki_cal_stride, dtype, device);
                        int64_t kvi_length = tlength - first_step;
                        int64_t kvi_beg = first_step % max_seq_len;
                        int64_t kvi_end = tlength % max_seq_len;
                        int64_t n_maxseqlen = kvi_length / max_seq_len;
                        std::vector<diopiConstTensorHandle_t> cat_kcal;
                        // assert(kvi_length < max_seq_len);
                        if (n_maxseqlen > 0) {
                            diopiTensorHandle_t catki_cache_beg;
                            shape[0] = local_kv_head_num;
                            shape[1] = max_seq_len - kvi_end;
                            shape[2] = size_per_head;
                            diopiRequireTensor(ctx, &catki_cache_beg, &newshape, nullptr, dtype, device);
                            diopiSlice(ctx, catki_cache_beg, k_cache, 1, kvi_end, max_seq_len, 1);
                            diopiTensorHandle_t catki_cache_end;
                            shape[0] = local_kv_head_num;
                            shape[1] = kvi_end;
                            shape[2] = size_per_head;
                            diopiRequireTensor(ctx, &catki_cache_end, &newshape, nullptr, dtype, device);
                            diopiSlice(ctx, catki_cache_end, k_cache, 1, 0, kvi_end, 1);
                            while (n_maxseqlen > 0) {
                                cat_kcal.emplace_back(catki_cache_beg);
                                cat_kcal.emplace_back(catki_cache_end);
                                --n_maxseqlen;
                            }
                        }
                        if (kvi_beg > kvi_end) {
                            diopiTensorHandle_t catki_cache_beg_gt;
                            shape[0] = local_kv_head_num;
                            shape[1] = max_seq_len - kvi_beg;
                            shape[2] = size_per_head;
                            char* catki_cache_beg_gt_ptr =
                                reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1);
                            diopiSize_t catki_cache_beg_gt_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(catki_cache_beg_gt_ptr)), -1};
                            diopiRequireTensor(ctx, &catki_cache_beg_gt, &newshape, &catki_cache_beg_gt_stride, dtype, device);
                            diopiSlice(ctx, catki_cache_beg_gt, k_cache, 1, kvi_beg, max_seq_len, 1);
                            cat_kcal.emplace_back(catki_cache_beg_gt);
                            if (kvi_end > 0) {
                                diopiTensorHandle_t catki_cache_end_gt;
                                shape[0] = local_kv_head_num;
                                shape[1] = kvi_end;
                                shape[2] = size_per_head;
                                char* catki_cache_end_gt_ptr =
                                    reinterpret_cast<char*>(catki_cache_beg_gt_ptr) + itemsize * local_kv_head_num * size_per_head * (max_seq_len - kvi_beg);
                                diopiSize_t catki_cache_end_gt_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(catki_cache_end_gt_ptr)), -1};
                                diopiRequireTensor(ctx, &catki_cache_end_gt, &newshape, &catki_cache_end_gt_stride, dtype, device);
                                diopiSlice(ctx, catki_cache_end_gt, k_cache, 1, 0, kvi_end, 1);
                                cat_kcal.emplace_back(catki_cache_end_gt);
                            }
                        } else if (kvi_beg < kvi_end) {
                            diopiTensorHandle_t catki_cache_end_lt;
                            shape[0] = local_kv_head_num;
                            shape[1] = kvi_end - kvi_beg;
                            shape[2] = size_per_head;
                            char* catki_cache_end_lt_ptr =
                                reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1);
                            diopiSize_t catki_cache_end_lt_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(catki_cache_end_lt_ptr)), -1};
                            diopiRequireTensor(ctx, &catki_cache_end_lt, &newshape, &catki_cache_end_lt_stride, dtype, device);
                            diopiSlice(ctx, catki_cache_end_lt, k_cache, 1, kvi_beg, kvi_end, 1);
                            cat_kcal.emplace_back(catki_cache_end_lt);
                        }
                        cat_kcal.emplace_back(ki_buffer1);
                        diopiTensorHandle_t ki_cal021;
                        newshape.len = 3;
                        shape[0] = local_head_num;
                        shape[1] = tlength - first_step + 1;
                        shape[2] = size_per_head;
                        char* ki_cal021_ptr =
                            reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1) * 2;
                        diopiSize_t ki_cal021_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(ki_cal021_ptr)), -1};
                        diopiRequireTensor(ctx, &ki_cal021, &newshape, &ki_cal021_stride, dtype, device);
                        combAsCat(ctx, ki_cal021, cat_kcal.data(), cat_kcal.size(), 1);
                        diopiPermute(ctx, ki_cal, ki_cal021, trans021);
                        // qk_cal
                        diopiTensorHandle_t qki_cal;
                        newshape.len = 3;
                        shape[0] = local_head_num;
                        shape[1] = 1;
                        shape[2] = tlength - first_step + 1;
                        char* qki_cal_ptr = reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1);
                        diopiSize_t qki_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qki_cal_ptr)), -1};
                        diopiRequireTensor(ctx, &qki_cal, &newshape, &qki_cal_stride, dtype, device);
                        diopiBmm(ctx, qki_cal, qi_cal, ki_cal);
                        diopiMulInpScalar(ctx, qki_cal, &inv_sqrt_dh);
                        // qk softmax, using ki_cal as buffer
                        diopiTensorHandle_t qki_softmax;
                        char* qki_softmax_ptr = reinterpret_cast<char*>(ki_cal_ptr);
                        diopiSize_t qki_softmax_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(qki_softmax_ptr)), -1};
                        diopiRequireTensor(ctx, &qki_softmax, &newshape, &qki_softmax_stride, dtype, device);
                        diopiSoftmax(ctx, qki_softmax, qki_cal, 2);
                        // vi_buffer
                        diopiTensorHandle_t vi_buffer;
                        newshape.len = 3;
                        shape[0] = local_kv_head_num;
                        shape[1] = 1;
                        shape[2] = size_per_head;
                        char* vi_buffer_ptr = reinterpret_cast<char*>(v_buffer_ptr) + i * itemsize * local_kv_head_num * size_per_head;
                        diopiSize_t vi_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(vi_buffer_ptr)), -1};
                        diopiRequireTensor(ctx, &vi_buffer, &newshape, &vi_buffer_stride, dtype, device);
                        // v cache
                        diopiTensorHandle_t v_cache;
                        newshape.len = 3;
                        shape[0] = local_kv_head_num;
                        shape[1] = max_seq_len;
                        shape[2] = size_per_head;
                        void* value_cache_ptr;
                        diopiGetTensorData(value_cache[i], &value_cache_ptr);
                        char* v_cache_ptr = reinterpret_cast<char*>(value_cache_ptr) + kv_cache_layer_offset;
                        diopiSize_t v_cache_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(v_cache_ptr)), -1};
                        diopiRequireTensor(ctx, &v_cache, &newshape, &v_cache_stride, dtype, device);
                        // update v cache, index is tlength_circ
                        diopiConstTensorHandle_t cat_vcache[3];
                        int64_t cat_vcache_length = 1;
                        if (tlength_circ > 0) {
                            diopiTensorHandle_t vi_cache_beg;
                            shape[0] = local_kv_head_num;
                            shape[1] = tlength_circ;
                            shape[2] = size_per_head;
                            char* vi_cache_beg_ptr =
                                reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1);
                            diopiSize_t vi_cache_beg_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(vi_cache_beg_ptr)), -1};
                            diopiRequireTensor(ctx, &vi_cache_beg, &newshape, &vi_cache_beg_stride, dtype, device);
                            diopiSlice(ctx, vi_cache_beg, v_cache, 1, 0, tlength_circ, 1);
                            cat_vcache[0] = vi_cache_beg;
                            cat_vcache[1] = vi_buffer;
                            cat_vcache_length += 1;
                        } else {
                            cat_vcache[0] = vi_buffer;
                        }
                        if (tlength_circ != max_seq_len - 1) {
                            diopiTensorHandle_t vi_cache_end;
                            shape[0] = local_kv_head_num;
                            shape[1] = max_seq_len - tlength_circ - 1;
                            shape[2] = size_per_head;
                            char* vi_cache_end_ptr =
                                reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1) * 2;
                            diopiSize_t vi_cache_end_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(vi_cache_end_ptr)), -1};
                            diopiRequireTensor(ctx, &vi_cache_end, &newshape, &vi_cache_end_stride, dtype, device);
                            diopiSlice(ctx, vi_cache_end, v_cache, 1, tlength_circ + 1, max_seq_len, 1);
                            cat_vcache[cat_vcache_length] = vi_cache_end;
                            cat_vcache_length += 1;
                        }
                        combAsCat(ctx, v_cache, cat_vcache, cat_vcache_length, 1);
                        // vi for cal
                        diopiTensorHandle_t vi_cal;
                        newshape.len = 3;
                        shape[0] = local_head_num;
                        shape[1] = tlength - first_step + 1;
                        shape[2] = size_per_head;
                        char* vi_cal_ptr = reinterpret_cast<char*>(workspace1_ptr) + itemsize * local_head_num * (tlength - first_step + 1);
                        diopiSize_t vi_cal_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(vi_cal_ptr)), -1};
                        diopiRequireTensor(ctx, &vi_cal, &newshape, &vi_cal_stride, dtype, device);
                        n_maxseqlen = kvi_length / max_seq_len;
                        std::vector<diopiConstTensorHandle_t> cat_vcal;
                        if (n_maxseqlen > 0) {
                            diopiTensorHandle_t catvi_cache_beg;
                            shape[0] = local_kv_head_num;
                            shape[1] = max_seq_len - kvi_end;
                            shape[2] = size_per_head;
                            diopiRequireTensor(ctx, &catvi_cache_beg, &newshape, nullptr, dtype, device);
                            diopiSlice(ctx, catvi_cache_beg, v_cache, 1, kvi_beg, max_seq_len, 1);
                            diopiTensorHandle_t catvi_cache_end;
                            shape[0] = local_kv_head_num;
                            shape[1] = kvi_end;
                            shape[2] = size_per_head;
                            diopiRequireTensor(ctx, &catvi_cache_end, &newshape, nullptr, dtype, device);
                            diopiSlice(ctx, catvi_cache_end, v_cache, 1, 0, kvi_beg, 1);
                            while (n_maxseqlen > 0) {
                                cat_vcal.emplace_back(catvi_cache_beg);
                                cat_vcal.emplace_back(catvi_cache_end);
                                --n_maxseqlen;
                            }
                        }
                        if (kvi_beg > kvi_end) {
                            diopiTensorHandle_t catvi_cache_beg_gt;
                            shape[0] = local_kv_head_num;
                            shape[1] = max_seq_len - kvi_beg;
                            shape[2] = size_per_head;
                            char* catvi_cache_beg_gt_ptr =
                                reinterpret_cast<char*>(vi_cal_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1);
                            diopiSize_t catvi_cache_beg_gt_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(catvi_cache_beg_gt_ptr)), -1};
                            diopiRequireTensor(ctx, &catvi_cache_beg_gt, &newshape, &catvi_cache_beg_gt_stride, dtype, device);
                            diopiSlice(ctx, catvi_cache_beg_gt, v_cache, 1, kvi_beg, max_seq_len, 1);
                            cat_kcal.emplace_back(catvi_cache_beg_gt);
                            if (kvi_end > 0) {
                                diopiTensorHandle_t catvi_cache_end_gt;
                                shape[0] = local_kv_head_num;
                                shape[1] = kvi_end;
                                shape[2] = size_per_head;
                                char* catvi_cache_end_gt_ptr =
                                    reinterpret_cast<char*>(catvi_cache_beg_gt_ptr) + itemsize * local_head_num * size_per_head * (max_seq_len - kvi_beg);
                                diopiSize_t catvi_cache_end_gt_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(catvi_cache_end_gt_ptr)), -1};
                                diopiRequireTensor(ctx, &catvi_cache_end_gt, &newshape, &catvi_cache_end_gt_stride, dtype, device);
                                diopiSlice(ctx, catvi_cache_end_gt, v_cache, 1, 0, kvi_end, 1);
                                cat_vcal.emplace_back(catvi_cache_end_gt);
                            }
                        } else if (kvi_beg < kvi_end) {
                            diopiTensorHandle_t catvi_cache_end_lt;
                            shape[0] = local_kv_head_num;
                            shape[1] = kvi_end - kvi_beg;
                            shape[2] = size_per_head;
                            char* catvi_cache_end_lt_ptr =
                                reinterpret_cast<char*>(vi_cal_ptr) + itemsize * local_head_num * size_per_head * (tlength - first_step + 1);
                            diopiSize_t catvi_cache_end_lt_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(catvi_cache_end_lt_ptr)), -1};
                            diopiRequireTensor(ctx, &catvi_cache_end_lt, &newshape, &catvi_cache_end_lt_stride, dtype, device);
                            diopiSlice(ctx, catvi_cache_end_lt, v_cache, 1, kvi_beg, kvi_end, 1);
                            cat_vcal.emplace_back(catvi_cache_end_lt);
                        }
                        cat_vcal.emplace_back(vi_buffer);
                        combAsCat(ctx, vi_cal, cat_vcal.data(), cat_vcal.size(), 1);
                        // * v
                        diopiTensorHandle_t inouti;
                        shape[0] = local_head_num;
                        shape[1] = 1;
                        shape[2] = size_per_head;
                        newshape.len = 3;
                        char* inouti_ptr = reinterpret_cast<char*>(inout_ptr) + i * itemsize * local_head_num * size_per_head;
                        diopiSize_t inouti_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(inouti_ptr)), -1};
                        diopiRequireTensor(ctx, &inouti, &newshape, &inouti_stride, dtype, device);
                        diopiBmm(ctx, inouti, qki_softmax, vi_cal);
                    } else if (same) {
                        diopiTensorHandle_t q_buffer_back_one;
                        shape[0] = local_head_num;
                        shape[1] = size_per_head;
                        newshape.len = 2;
                        void* q_buffer_back_one_ptr;
                        diopiGetTensorData(q_buffer_back, &q_buffer_back_one_ptr);
                        diopiSize_t q_buffer_back_one_stride{
                            static_cast<const int64_t*>(
                                reinterpret_cast<int64_t*>(reinterpret_cast<char*>(q_buffer_back_one_ptr) + i * itemsize * local_head_num * size_per_head)),
                            -1};
                        diopiRequireTensor(ctx, &q_buffer_back_one, &newshape, &q_buffer_back_one_stride, dtype, device);
                        diopiTensorHandle_t q_buffer_one;
                        shape[0] = local_head_num;
                        shape[1] = size_per_head;
                        newshape.len = 2;
                        char* q_buffer_one_ptr = q_buffer_ptr + i * itemsize * local_head_num * size_per_head;
                        diopiSize_t q_buffer_one_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(q_buffer_one_ptr)), -1};
                        diopiRequireTensor(ctx, &q_buffer_one, &newshape, &q_buffer_one_stride, dtype, device);
                        diopiLmdeployCopyD2D(ctx, q_buffer_one, q_buffer_back_one, true);
                    }
                } else {
                    return diopiErrorOccurred;
                }
            }
        }
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}

DIOPI_API diopiError_t diopiSetupTopkRuntimeArgsInp(diopiContextHandle_t ctx, diopiTensorHandle_t top_ks, diopiTensorHandle_t top_ps,
                                                    diopiTensorHandle_t skip_decode, int64_t batch_size, int64_t top_k, int64_t top_ks_size, float top_p,
                                                    int64_t top_ps_size) {
    int64_t TOP_K_MAX = 1024;

    diopiDtype_t intdtype;
    diopiGetTensorDtype(top_ks, &intdtype);
    diopiDtype_t dtype;
    diopiGetTensorDtype(top_ps, &dtype);
    int64_t intitemsize = -1;
    diopiGetTensorElemSize(top_ks, &intitemsize);
    int64_t itemsize = -1;
    diopiGetTensorElemSize(top_ps, &itemsize);

    std::vector<int64_t> shape{batch_size};
    diopiSize_t newshape{shape.data(), 1};

    diopiTensorHandle_t h_top_ks;
    void* h_top_ks_data;
    diopiRequireTensor(ctx, &h_top_ks, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
    diopiLmdeployCopyD2H(ctx, h_top_ks, top_ks, false);
    diopiGetTensorData(h_top_ks, &h_top_ks_data);
    diopiTensorHandle_t forcast = nullptr;
    diopiTensorHandle_t h_top_ps;
    float h_top_ps_data[batch_size];
    diopiSize_t h_top_ps_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_top_ps_data)), -1};
    diopiRequireTensor(ctx, &h_top_ps, &newshape, &h_top_ps_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
    if (dtype != diopiDtype_t::diopi_dtype_float32) {
        diopiRequireTensor(ctx, &forcast, &newshape, nullptr, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_device);
        diopiCastDtype(ctx, forcast, top_ps);
        diopiLmdeployCopyD2H(ctx, h_top_ps, forcast, false);
    } else {
        diopiLmdeployCopyD2H(ctx, h_top_ps, top_ps, false);
    }
    diopiTensorHandle_t h_skip_decode;
    bool h_skip_decode_data[batch_size];
    diopiSize_t h_skip_decode_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_skip_decode_data)), -1};
    diopiRequireTensor(ctx, &h_skip_decode, &newshape, &h_skip_decode_stride, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
    diopiLmdeployCopyD2H(ctx, h_skip_decode, skip_decode, false);

    for (int64_t i = 0; i < batch_size; i++) {
        int64_t h_top_ks_i =
            intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(h_top_ks_data) + i) : *(reinterpret_cast<int64_t*>(h_top_ks_data) + i);
        int64_t k = top_ks_size > 1 ? h_top_ks_i : top_k;
        float p = top_ps_size > 1 ? h_top_ps_data[i] : top_p;
        if (k == 0 && p == 0.0f) {
            // FT's topp implementation does not support topp = 0.0f, but it equivalent to greedy search.
            // So, we set the topk = 1 as an alternative solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f) {
            // for compatibility <= FT5.0.
            // This case corresponds to the old topk sampling, which is equivalent to
            // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
            // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
            // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
            // compatibility.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX=64.
        h_top_ks_i = k > TOP_K_MAX ? TOP_K_MAX : k;
        if (k > TOP_K_MAX) {
            printf(
                "[WARNING] topk (%d) is larger than max supported number (%d) for token %d"
                " clip to max supported number %d. \n",
                k,
                TOP_K_MAX,
                i,
                h_top_ks_i);
        }
        if (intdtype == diopiDtype_t::diopi_dtype_int32) {
            int32_t* h_top_ks_i_ptr = reinterpret_cast<int32_t*>(h_top_ks_data) + i;
            *h_top_ks_i_ptr = h_top_ks_i;
        } else {
            int64_t* h_top_ks_i_ptr = reinterpret_cast<int64_t*>(h_top_ks_data) + i;
            *h_top_ks_i_ptr = h_top_ks_i;
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        h_top_ps_data[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf(
                "[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                " clip to closest number %f.\n",
                p,
                i,
                h_top_ps_data[i]);
        }
        h_skip_decode_data[i] = k == 0;
    }

    diopiLmdeployCopyH2D(ctx, top_ks, h_top_ks, false);
    if (dtype != diopiDtype_t::diopi_dtype_float32) {
        diopiLmdeployCopyH2D(ctx, forcast, h_top_ps, false);
        diopiCastDtype(ctx, top_ps, forcast);
    } else {
        diopiLmdeployCopyH2D(ctx, top_ps, h_top_ps, false);
    }
    diopiLmdeployCopyH2D(ctx, skip_decode, h_skip_decode, false);  // check? out savenpy
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSetupToppRuntimeArgsInp(diopiContextHandle_t ctx, diopiTensorHandle_t top_ks, diopiTensorHandle_t top_ps,
                                                    diopiTensorHandle_t skip_decode, int64_t batch_size, int64_t top_k, int64_t top_ks_size, float top_p,
                                                    int64_t top_ps_size, diopiTensorHandle_t initial_top_p_buf, diopiTensorHandle_t top_p_decay_buf,
                                                    diopiConstTensorHandle_t top_p_decay, diopiTensorHandle_t top_p_min_buf, diopiConstTensorHandle_t top_p_min,
                                                    diopiTensorHandle_t top_p_reset_ids_buf, diopiConstTensorHandle_t top_p_reset_ids) {
    diopiDtype_t intdtype;
    diopiGetTensorDtype(top_ks, &intdtype);
    diopiDtype_t dtype;
    diopiGetTensorDtype(top_ps, &dtype);

    std::vector<int64_t> shape{batch_size};
    diopiSize_t newshape{shape.data(), 1};

    diopiTensorHandle_t h_top_ks;
    void* h_top_ks_data;
    diopiRequireTensor(ctx, &h_top_ks, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
    diopiLmdeployCopyD2H(ctx, h_top_ks, top_ks, false);
    diopiGetTensorData(h_top_ks, &h_top_ks_data);
    diopiTensorHandle_t forcast = nullptr;
    diopiTensorHandle_t h_top_ps;
    float h_top_ps_data[batch_size];
    diopiSize_t h_top_ps_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_top_ps_data)), -1};
    diopiRequireTensor(ctx, &h_top_ps, &newshape, &h_top_ps_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
    if (dtype != diopiDtype_t::diopi_dtype_float32) {
        diopiRequireTensor(ctx, &forcast, &newshape, nullptr, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_device);
        diopiCastDtype(ctx, forcast, top_ps);
        diopiLmdeployCopyD2H(ctx, h_top_ps, forcast, false);
    } else {
        diopiLmdeployCopyD2H(ctx, h_top_ps, top_ps, false);
    }
    diopiTensorHandle_t h_skip_decode;
    bool h_skip_decode_data[batch_size];
    diopiSize_t h_skip_decode_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_skip_decode_data)), -1};
    diopiRequireTensor(ctx, &h_skip_decode, &newshape, &h_skip_decode_stride, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
    diopiLmdeployCopyD2H(ctx, h_skip_decode, skip_decode, false);
    diopiTensorHandle_t h_top_p_decay_buf;
    float h_top_p_decay_buf_data[batch_size];
    if (top_p_decay == nullptr) {
        diopiScalar_t scalar_done{dtype, double(1)};
        diopiFill(ctx, top_p_decay_buf, &scalar_done);
    } else {
        diopiSize_t h_top_p_decay_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_top_p_decay_buf_data)), -1};
        diopiRequireTensor(ctx, &h_top_p_decay_buf, &newshape, &h_top_p_decay_buf_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiCastDtype(ctx, forcast, top_p_decay);
            diopiLmdeployCopyD2H(ctx, h_top_p_decay_buf, forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_top_p_decay_buf, top_p_decay, false);
        }
    }
    diopiTensorHandle_t h_top_p_min_buf;
    float h_top_p_min_buf_data[batch_size];
    if (top_p_min == nullptr) {
        diopiScalar_t scalar_d001{dtype, double(1e-6)};
        diopiFill(ctx, top_p_min_buf, &scalar_d001);
    } else {
        diopiSize_t h_top_p_min_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_top_p_min_buf_data)), -1};
        diopiRequireTensor(ctx, &h_top_p_min_buf, &newshape, &h_top_p_min_buf_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiCastDtype(ctx, forcast, top_p_min);
            diopiLmdeployCopyD2H(ctx, h_top_p_min_buf, forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_top_p_min_buf, top_p_min, false);
        }
    }

    if (top_p_reset_ids == nullptr) {
        diopiScalar_t scalar_i_one{diopiDtype_t::diopi_dtype_float64, double(-1)};
        diopiFill(ctx, top_p_reset_ids_buf, &scalar_i_one);
    } else {
        diopiLmdeployCopyD2D(ctx, top_p_reset_ids_buf, top_p_reset_ids, false);
    }

    for (int64_t i = 0; i < batch_size; i++) {
        int64_t h_top_ks_i =
            intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(h_top_ks_data) + i) : *(reinterpret_cast<int64_t*>(h_top_ks_data) + i);
        int64_t k = top_ks_size > 1 ? h_top_ks_i : top_k;
        float p = top_ps_size > 1 ? h_top_ps_data[i] : top_p;
        if (k == 0 && p == 0.0f) {
            // FT's topp implementation does not support topp = 0.0f, but it equivalent to greedy search.
            // So, we set the topk = 1 as an alternative solution.
            k = 1;
        }
        if (intdtype == diopiDtype_t::diopi_dtype_int32) {
            int32_t* h_top_ks_i_ptr = reinterpret_cast<int32_t*>(h_top_ks_data) + i;
            *h_top_ks_i_ptr = k;
        } else {
            int64_t* h_top_ks_i_ptr = reinterpret_cast<int64_t*>(h_top_ks_data) + i;
            *h_top_ks_i_ptr = k;
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        h_top_ps_data[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf(
                "[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                " clip to closest number %f.\n",
                p,
                i,
                h_top_ps_data[i]);
        }
        h_skip_decode_data[i] = k > 0;

        if (top_p_decay != nullptr && (h_top_p_decay_buf_data[i] > 1.0f || h_top_p_decay_buf_data[i] <= 0.0f)) {  // default 1.0f
            printf(
                "[WARNING] top_p_decay_buf (%f) is out of range ([0.0, 1.0f]) for token %d,"
                " change to 1.0f.\n",
                h_top_p_decay_buf_data[i],
                i);
            h_top_p_decay_buf_data[i] = 1.0f;
        }
        if (top_p_min != nullptr && (h_top_p_min_buf_data[i] > 1.0f || h_top_p_min_buf_data[i] <= 0.0f)) {  // default 1e-6f;
            printf(
                "[WARNING] top_p_min_buf (%f) is out of range ([0.0, 1.0f]) for token %d,"
                " change to 0.5f.\n",
                h_top_p_min_buf_data[i],
                i);
            h_top_p_min_buf_data[i] = 0.5f;
        }
    }

    diopiLmdeployCopyH2D(ctx, skip_decode, h_skip_decode, false);
    diopiLmdeployCopyH2D(ctx, top_ks, h_top_ks, false);
    if (dtype != diopiDtype_t::diopi_dtype_float32) {
        diopiLmdeployCopyH2D(ctx, forcast, h_top_ps, false);
        diopiCastDtype(ctx, top_ps, forcast);
        if (top_p_decay != nullptr) {
            diopiLmdeployCopyH2D(ctx, forcast, h_top_p_decay_buf, false);
            diopiCastDtype(ctx, top_p_decay_buf, forcast);
        }
        if (top_p_min != nullptr) {
            diopiLmdeployCopyH2D(ctx, forcast, h_top_p_min_buf, false);
            diopiCastDtype(ctx, top_p_min_buf, forcast);
        }
    } else {
        diopiLmdeployCopyH2D(ctx, top_ps, h_top_ps, false);
        if (top_p_decay != nullptr) {
            diopiLmdeployCopyH2D(ctx, top_p_decay_buf, h_top_p_decay_buf, false);
        }
        if (top_p_min != nullptr) {
            diopiLmdeployCopyH2D(ctx, top_p_min_buf, h_top_p_min_buf, false);
        }
    }
    diopiLmdeployCopyD2D(ctx, initial_top_p_buf, top_ps, false);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTopKSampling(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiTensorHandle_t logits, diopiTensorHandle_t workspace,
                                         int64_t* workspace_size, int64_t fusion_level, diopiConstTensorHandle_t end_ids, diopiTensorHandle_t finished,
                                         diopiTensorHandle_t sequence_lengths, int64_t step, int64_t batch_size, int64_t vocab_size_padded,
                                         diopiConstTensorHandle_t runtime_top_k, diopiConstTensorHandle_t runtime_top_p, diopiConstTensorHandle_t skip_decode,
                                         diopiTensorHandle_t cum_log_probs, diopiTensorHandle_t output_log_probs, diopiGeneratorHandle_t* generators) {
    if (fusion_level >= 0) {
        // vocab_size_padded == vocab_size in llamav2.cc
        // workspace_size
        diopiSize_t shapeinfo;
        diopiGetTensorShape(logits, &shapeinfo);
        int64_t intitemsize = -1;
        diopiGetTensorElemSize(output_ids, &intitemsize);
        int64_t itemsize = -1;
        diopiGetTensorElemSize(logits, &itemsize);
        if (*workspace_size < 0) {
            *workspace_size = batch_size * vocab_size_padded * itemsize +  // forcast/logits_forsample
                              vocab_size_padded * itemsize +               // topk_value
                              vocab_size_padded * sizeof(int64_t) +        // topk_index
                              0;
            return diopiSuccess;
        }  // TODO: check logits and max k

        assert(shapeinfo.data[1] == vocab_size_padded);

        std::vector<int64_t> shape(4);
        diopiSize_t newshape{shape.data(), 4};
        void* logits_ptr;
        diopiGetTensorData(logits, &logits_ptr);
        void* output_ids_ptr;
        diopiGetTensorData(output_ids, &output_ids_ptr);
        void* workspace_ptr;
        diopiGetTensorData(workspace, &workspace_ptr);
        diopiDtype_t intdtype;
        diopiGetTensorDtype(output_ids, &intdtype);
        void* inout_ptr;
        diopiGetTensorData(logits, &inout_ptr);
        diopiDevice_t device;
        diopiGetTensorDevice(logits, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(logits, &dtype);

        diopiTensorHandle_t h_finished = nullptr;
        bool* h_finished_data = nullptr;
        if (finished != nullptr) {
            shape[0] = batch_size;
            newshape.len = 1;
            diopiRequireTensor(ctx, &h_finished, &newshape, nullptr, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_finished, finished, false);
            diopiGetTensorData(h_finished, reinterpret_cast<void**>(&h_finished_data));
        }
        diopiTensorHandle_t h_sequence_lengths = nullptr;
        void* h_sequence_lengths_data = nullptr;
        if (sequence_lengths != nullptr) {
            shape[0] = batch_size;
            newshape.len = 1;
            diopiRequireTensor(ctx, &h_sequence_lengths, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_sequence_lengths, sequence_lengths, false);
            diopiGetTensorData(h_sequence_lengths, &h_sequence_lengths_data);
        }
        diopiTensorHandle_t h_end_ids;
        void* h_end_ids_data;
        shape[0] = batch_size;
        newshape.len = 1;
        diopiRequireTensor(ctx, &h_end_ids, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
        diopiGetTensorData(h_end_ids, &h_end_ids_data);
        diopiLmdeployCopyD2H(ctx, h_end_ids, end_ids, false);
        diopiTensorHandle_t h_skip_decode = nullptr;
        bool* h_skip_decode_data = nullptr;
        if (skip_decode != nullptr) {
            shape[0] = batch_size;
            newshape.len = 1;
            diopiRequireTensor(ctx, &h_skip_decode, &newshape, nullptr, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_skip_decode, skip_decode, false);
            diopiGetTensorData(h_skip_decode, reinterpret_cast<void**>(&h_skip_decode_data));
        }
        diopiTensorHandle_t h_runtime_top_k;
        void* h_runtime_top_k_data;
        diopiRequireTensor(ctx, &h_runtime_top_k, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
        diopiGetTensorData(h_runtime_top_k, &h_runtime_top_k_data);
        diopiLmdeployCopyD2H(ctx, h_runtime_top_k, runtime_top_k, false);
        diopiTensorHandle_t h_runtime_top_p;
        float h_runtime_top_p_data[batch_size];
        diopiSize_t h_runtime_top_p_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_runtime_top_p_data)), -1};
        diopiRequireTensor(ctx, &h_runtime_top_p, &newshape, &h_runtime_top_p_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiTensorHandle_t runtime_top_p_forcast;
            char* runtime_top_p_forcast_ptr = reinterpret_cast<char*>(workspace_ptr);
            diopiSize_t runtime_top_p_forcast_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(runtime_top_p_forcast_ptr)), -1};
            diopiRequireTensor(
                ctx, &runtime_top_p_forcast, &newshape, &runtime_top_p_forcast_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_device);
            diopiCastDtype(ctx, runtime_top_p_forcast, runtime_top_p);
            diopiLmdeployCopyD2H(ctx, h_runtime_top_p, runtime_top_p_forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_runtime_top_p, runtime_top_p, false);
        }

        // addBiasEndMask
        double MAX_T_VAL = dtype == diopiDtype_t::diopi_dtype_float16 ? 65504.F : FLT_MAX;
        diopiScalar_t scalar_dmax{dtype, MAX_T_VAL};
        diopiScalar_t scalar_dmin{dtype, -MAX_T_VAL};
        for (int64_t i = 0; i < batch_size; i++) {
            // no bias and vocab_size_padded = vocab_size
            if (h_finished_data != nullptr && h_finished_data[i]) {
                int64_t end_id = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(h_end_ids_data) + i)
                                                                             : *(reinterpret_cast<int64_t*>(h_end_ids_data) + i);
                if (end_id < vocab_size_padded) {
                    diopiTensorHandle_t logit_masks;
                    newshape.len = 1;
                    shape[0] = vocab_size_padded;
                    char* logit_masks_ptr = reinterpret_cast<char*>(logits_ptr) + i * itemsize * vocab_size_padded;
                    diopiSize_t logit_masks_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_masks_ptr)), -1};
                    diopiRequireTensor(ctx, &logit_masks, &newshape, &logit_masks_stride, dtype, device);
                    diopiFill(ctx, logit_masks, &scalar_dmin);
                    diopiTensorHandle_t logit_mask;
                    newshape.len = 1;
                    shape[0] = 1;
                    char* logit_mask_ptr = reinterpret_cast<char*>(logits_ptr) + i * itemsize * vocab_size_padded + itemsize * end_id;
                    diopiSize_t logit_mask_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_mask_ptr)), -1};
                    diopiRequireTensor(ctx, &logit_mask, &newshape, &logit_mask_stride, dtype, device);
                    diopiFill(ctx, logit_mask, &scalar_dmax);
                }
            }
        }

        // addBiasSoftMax. no bias and vocab_size_padded = vocab_size with norm softmax
        diopiTensorHandle_t logits_forsample;
        if (cum_log_probs != nullptr || output_log_probs != nullptr) {
            newshape.len = 2;
            shape[0] = batch_size;
            shape[1] = vocab_size_padded;
            char* logits_forsample_ptr = reinterpret_cast<char*>(workspace_ptr);
            diopiSize_t logits_forsample_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logits_forsample_ptr)), -1};
            diopiRequireTensor(ctx, &logits_forsample, &newshape, &logits_forsample_stride, dtype, device);
            diopiSoftmax(ctx, logits_forsample, logits, 1);
        } else {
            logits_forsample = logits;
        }
        void* logits_forsample_ptr;
        diopiGetTensorData(logits_forsample, &logits_forsample_ptr);

        // BatchTopKSampling
        diopiScalar_t scalar_did{dtype, double(0)};
        diopiScalar_t scalar_done{dtype, double(1)};
        diopiTensorHandle_t topk_value_buffer;
        newshape.len = 1;
        shape[0] = vocab_size_padded;
        char* topk_value_buffer_ptr = reinterpret_cast<char*>(workspace_ptr) + itemsize * batch_size * vocab_size_padded;
        diopiSize_t topk_value_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topk_value_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &topk_value_buffer, &newshape, &topk_value_buffer_stride, dtype, device);
        diopiTensorHandle_t topk_index_buffer;
        newshape.len = 1;
        shape[0] = vocab_size_padded;
        char* topk_index_buffer_ptr = reinterpret_cast<char*>(topk_value_buffer_ptr) + itemsize * vocab_size_padded;
        diopiSize_t topk_index_buffer_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topk_index_buffer_ptr)), -1};
        diopiRequireTensor(ctx, &topk_index_buffer, &newshape, &topk_index_buffer_stride, diopiDtype_t::diopi_dtype_int64, device);
        for (int64_t i = 0; i < batch_size; i++) {
            if (h_skip_decode_data != nullptr && h_skip_decode_data[i]) continue;
            diopiTensorHandle_t output_id;
            newshape.len = 1;
            shape[0] = 1;
            char* output_id_ptr = reinterpret_cast<char*>(output_ids_ptr) + step * intitemsize * batch_size + i * intitemsize;
            diopiSize_t output_id_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(output_id_ptr)), -1};
            diopiRequireTensor(ctx, &output_id, &newshape, &output_id_stride, intdtype, device);
            if (h_finished_data != nullptr && h_finished_data[i]) {
                scalar_did.fval = double(diopiDtype_t::diopi_dtype_int32 == intdtype ? *(reinterpret_cast<int32_t*>(h_end_ids_data) + i)
                                                                                     : *(reinterpret_cast<int64_t*>(h_end_ids_data) + i));
                diopiFill(ctx, output_id, &scalar_did);
                continue;
            }
            int64_t k = diopiDtype_t::diopi_dtype_int32 == intdtype ? *(reinterpret_cast<int32_t*>(h_runtime_top_k_data) + i)
                                                                    : *(reinterpret_cast<int64_t*>(h_runtime_top_k_data) + i);
            float p{h_runtime_top_p_data[i]};
            diopiTensorHandle_t logit;
            newshape.len = 1;
            shape[0] = vocab_size_padded;
            char* logit_ptr = reinterpret_cast<char*>(logits_forsample_ptr) + i * itemsize * vocab_size_padded;
            diopiSize_t logit_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_ptr)), -1};
            diopiRequireTensor(ctx, &logit, &newshape, &logit_stride, dtype, device);
            diopiTensorHandle_t topk_value;
            newshape.len = 1;
            shape[0] = k;
            diopiRequireTensor(ctx, &topk_value, &newshape, &topk_value_buffer_stride, dtype, device);
            diopiTensorHandle_t topk_index;
            newshape.len = 1;
            shape[0] = k;
            diopiRequireTensor(ctx, &topk_index, &newshape, &topk_index_buffer_stride, diopiDtype_t::diopi_dtype_int64, device);
            diopiTopk(ctx, topk_value, topk_index, logit, k, 0, true, true);
            // return diopiSuccess;

            if (cum_log_probs == nullptr && output_log_probs == nullptr) {
                diopiTensorHandle_t topk_max;
                newshape.len = 1;
                shape[0] = 1;
                char* topk_max_ptr = reinterpret_cast<char*>(logit_ptr);
                diopiSize_t topk_max_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topk_max_ptr)), -1};
                diopiRequireTensor(ctx, &topk_max, &newshape, &topk_max_stride, dtype, device);
                diopiMaxAll(ctx, topk_max, topk_value);
                diopiSubInp(ctx, topk_value, topk_max, &scalar_done);
                diopiExpInp(ctx, topk_value);
            }

            diopiTensorHandle_t rand_num;
            newshape.len = 1;
            shape[0] = 1;
            char* rand_num_ptr = reinterpret_cast<char*>(logit_ptr);
            diopiSize_t rand_num_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(rand_num_ptr)), -1};
            diopiRequireTensor(ctx, &rand_num, &newshape, &rand_num_stride, diopiDtype_t::diopi_dtype_float32, device);
            diopiUniformInp(ctx, rand_num, 0, 1, generators[i]);
            diopiTensorHandle_t h_rand_num;
            float h_rand_num_data[1];
            diopiSize_t h_rand_num_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_rand_num_data)), -1};
            diopiRequireTensor(ctx, &h_rand_num, &newshape, &h_rand_num_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_rand_num, rand_num, false);
            h_rand_num_data[0] = 0.5;  // for test

            diopiTensorHandle_t topk_sum;
            char* topk_sum_ptr = reinterpret_cast<char*>(logit_ptr);
            diopiSize_t topk_sum_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topk_sum_ptr)), -1};
            diopiRequireTensor(ctx, &topk_sum, &newshape, &topk_sum_stride, dtype, device);
            newshape.len = 1;
            shape[0] = 0;
            diopiSum(ctx, topk_sum, topk_value, newshape);
            diopiTensorHandle_t h_topk_sum;
            newshape.len = 1;
            shape[0] = 1;
            float h_topk_sum_data[1];
            diopiSize_t h_topk_sum_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_topk_sum_data)), -1};
            diopiRequireTensor(ctx, &h_topk_sum, &newshape, &h_topk_sum_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
            if (dtype != diopiDtype_t::diopi_dtype_float32) {
                diopiTensorHandle_t topk_sum_forcast;
                char* topk_sum_forcast_ptr = reinterpret_cast<char*>(topk_sum_ptr) + itemsize;
                diopiSize_t topk_sum_forcast_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topk_sum_forcast_ptr)), -1};
                diopiRequireTensor(ctx, &topk_sum_forcast, &newshape, &topk_sum_forcast_stride, diopiDtype_t::diopi_dtype_float32, device);
                diopiCastDtype(ctx, topk_sum_forcast, topk_sum);
                diopiLmdeployCopyD2H(ctx, h_topk_sum, topk_sum_forcast, false);
            } else {
                diopiLmdeployCopyD2H(ctx, h_topk_sum, topk_sum, false);
            }
            diopiTensorHandle_t h_topk_value;
            newshape.len = 1;
            shape[0] = k;
            float h_topk_value_data[k];
            diopiSize_t h_topk_value_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_topk_value_data)), -1};
            diopiRequireTensor(ctx, &h_topk_value, &newshape, &h_topk_value_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
            if (dtype != diopiDtype_t::diopi_dtype_float32) {
                diopiTensorHandle_t topk_value_forcast;
                char* topk_value_forcast_ptr = reinterpret_cast<char*>(topk_sum_ptr) + itemsize;
                assert(k + 1 < vocab_size_padded);
                diopiSize_t topk_value_forcast_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topk_value_forcast_ptr)), -1};
                diopiRequireTensor(ctx, &topk_value_forcast, &newshape, &topk_value_forcast_stride, diopiDtype_t::diopi_dtype_float32, device);
                diopiCastDtype(ctx, topk_value_forcast, topk_value);
                diopiLmdeployCopyD2H(ctx, h_topk_value, topk_value_forcast, false);
            } else {
                diopiLmdeployCopyD2H(ctx, h_topk_value, topk_value, false);
            }
            diopiTensorHandle_t h_topk_index;
            int64_t h_topk_index_data[k];
            diopiSize_t h_topk_index_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_topk_index_data)), -1};
            diopiRequireTensor(ctx, &h_topk_index, &newshape, &h_topk_index_stride, diopiDtype_t::diopi_dtype_int64, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_topk_index, topk_index, false);
            int64_t k_index = k - 1;
            h_rand_num_data[0] = h_rand_num_data[0] * p * h_topk_sum_data[0];
            for (int64_t ii = 0; ii < k - 1; ii++) {
                h_rand_num_data[0] -= h_topk_value_data[i];
                if (h_rand_num_data[0] <= 0) {
                    k_index = ii;
                    break;
                }
            }
            scalar_did.fval = double(h_topk_index_data[k_index]);
            diopiFill(ctx, output_id, &scalar_did);
            if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                diopiTensorHandle_t top_value;
                newshape.len = 1;
                shape[0] = 1;
                void* topk_value_ptr;
                diopiGetTensorData(topk_value, &topk_value_ptr);
                char* top_value_ptr = reinterpret_cast<char*>(topk_value_ptr) + itemsize * k_index;
                diopiSize_t top_value_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(top_value_ptr)), -1};
                diopiRequireTensor(ctx, &top_value, &newshape, &top_value_stride, dtype, device);
                diopiLogInp(ctx, top_value);
                if (cum_log_probs != nullptr) {
                    diopiTensorHandle_t cum_log_prob;
                    void* cum_log_probs_ptr;
                    diopiGetTensorData(cum_log_probs, &cum_log_probs_ptr);
                    float* cum_log_prob_ptr = reinterpret_cast<float*>(cum_log_probs_ptr) + i;
                    diopiSize_t cum_log_prob_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(cum_log_prob_ptr)), -1};
                    diopiRequireTensor(ctx, &cum_log_prob, &newshape, &cum_log_prob_stride, diopiDtype_t::diopi_dtype_float32, device);
                    if (dtype != diopiDtype_t::diopi_dtype_float32) {
                        diopiCastDtype(ctx, cum_log_prob, top_value);
                    } else {
                        diopiLmdeployCopyD2D(ctx, cum_log_prob, top_value, false);
                    }
                }
                if (output_log_probs != nullptr) {
                    diopiTensorHandle_t output_log_prob;
                    void* output_log_probs_ptr;
                    diopiGetTensorData(output_log_probs, &output_log_probs_ptr);
                    float* output_log_prob_ptr = reinterpret_cast<float*>(output_log_probs_ptr) + i;
                    diopiSize_t output_log_prob_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(output_log_prob_ptr)), -1};
                    diopiRequireTensor(ctx, &output_log_prob, &newshape, &output_log_prob_stride, diopiDtype_t::diopi_dtype_float32, device);
                    diopiLogInp(ctx, topk_sum);
                    diopiSub(ctx, output_log_prob, top_value, topk_sum, &scalar_done);
                }
            }
            if (sequence_lengths != nullptr && finished != nullptr) {
                if (!h_finished_data[i]) {
                    if (intdtype == diopiDtype_t::diopi_dtype_int32) {
                        int32_t* h_sequence_length_i = reinterpret_cast<int32_t*>(h_sequence_lengths_data) + i;
                        *h_sequence_length_i += 1;
                    } else {
                        int64_t* h_sequence_length_i = reinterpret_cast<int64_t*>(h_sequence_lengths_data) + i;
                        *h_sequence_length_i += 1;
                    }
                }
                int64_t h_end_ids_i = diopiDtype_t::diopi_dtype_int32 == intdtype ? *(reinterpret_cast<int32_t*>(h_end_ids_data) + i)
                                                                                  : *(reinterpret_cast<int64_t*>(h_end_ids_data) + i);
                h_finished_data[i] = int64_t(scalar_did.fval) == h_end_ids_i;
            }
        }
        if (sequence_lengths != nullptr && finished != nullptr) {
            diopiLmdeployCopyH2D(ctx, sequence_lengths, h_sequence_lengths, false);
            diopiLmdeployCopyH2D(ctx, finished, h_finished, false);
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTopPSampling(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiTensorHandle_t logits,
                                         diopiTensorHandle_t persistent_workspace, int64_t* persistent_workspace_size, diopiTensorHandle_t workspace,
                                         int64_t* workspace_size, int64_t fusion_level, diopiConstTensorHandle_t end_ids, diopiTensorHandle_t finished,
                                         diopiTensorHandle_t sequence_lengths, int64_t step, int64_t batch_size, int64_t vocab_size_padded,
                                         diopiTensorHandle_t runtime_top_p, diopiConstTensorHandle_t runtime_initial_top_p,
                                         diopiConstTensorHandle_t top_p_decay_buf, diopiConstTensorHandle_t top_p_min_buf,
                                         diopiConstTensorHandle_t top_p_reset_ids_buf, diopiConstTensorHandle_t skip_decode, diopiTensorHandle_t cum_log_probs,
                                         diopiTensorHandle_t output_log_probs, diopiGeneratorHandle_t* generators) {
    if (fusion_level >= 0) {
        // workspace_size
        diopiSize_t shapeinfo;
        diopiGetTensorShape(logits, &shapeinfo);
        int64_t intitemsize = -1;
        diopiGetTensorElemSize(output_ids, &intitemsize);
        int64_t itemsize = -1;
        diopiGetTensorElemSize(logits, &itemsize);
        if (*workspace_size < 0 || *persistent_workspace_size < 0) {
            *workspace_size = itemsize * batch_size * vocab_size_padded +  // logits_forsample
                              vocab_size_padded * sizeof(bool) +           // check
                              vocab_size_padded * itemsize * 2 +           // sort and cumsum
                              vocab_size_padded * sizeof(int64_t) +        // sort index
                              // itemsize + sizeof(float) + // rand_num
                              0;
            *persistent_workspace_size = intitemsize * batch_size * vocab_size_padded +  // topp_id_vals_buf
                                         intitemsize * (batch_size + 1) +                // topp_offset_buf
                                         intitemsize * (batch_size + 1) +                // begin_topp_offset_buf
                                         0;
            return diopiSuccess;
        }

        std::vector<int64_t> shape(4);
        diopiSize_t newshape{shape.data(), 4};
        void* logits_ptr;
        diopiGetTensorData(logits, &logits_ptr);
        void* output_ids_ptr;
        diopiGetTensorData(output_ids, &output_ids_ptr);
        void* workspace_ptr;
        diopiGetTensorData(workspace, &workspace_ptr);
        void* persistent_workspace_ptr;
        diopiGetTensorData(persistent_workspace, &persistent_workspace_ptr);
        diopiDtype_t intdtype;
        diopiGetTensorDtype(output_ids, &intdtype);
        diopiDevice_t device;
        diopiGetTensorDevice(logits, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(logits, &dtype);

        diopiTensorHandle_t h_finished = nullptr;
        bool* h_finished_data = nullptr;
        if (finished != nullptr) {
            shape[0] = batch_size;
            newshape.len = 1;
            diopiRequireTensor(ctx, &h_finished, &newshape, nullptr, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_finished, finished, false);
            diopiGetTensorData(h_finished, reinterpret_cast<void**>(&h_finished_data));
        }
        diopiTensorHandle_t h_sequence_lengths = nullptr;
        void* h_sequence_lengths_data = nullptr;
        if (sequence_lengths != nullptr) {
            shape[0] = batch_size;
            newshape.len = 1;
            diopiRequireTensor(ctx, &h_sequence_lengths, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_sequence_lengths, sequence_lengths, false);
            diopiGetTensorData(h_sequence_lengths, &h_sequence_lengths_data);
        }
        diopiTensorHandle_t h_end_ids;
        void* h_end_ids_data;
        diopiRequireTensor(ctx, &h_end_ids, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
        diopiGetTensorData(h_end_ids, &h_end_ids_data);
        diopiLmdeployCopyD2H(ctx, h_end_ids, end_ids, false);
        diopiTensorHandle_t h_skip_decode = nullptr;
        bool* h_skip_decode_data = nullptr;
        if (skip_decode != nullptr) {
            shape[0] = batch_size;
            newshape.len = 1;
            diopiRequireTensor(ctx, &h_skip_decode, &newshape, nullptr, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
            diopiLmdeployCopyD2H(ctx, h_skip_decode, skip_decode, false);
            diopiGetTensorData(h_skip_decode, reinterpret_cast<void**>(&h_skip_decode_data));
        }
        diopiTensorHandle_t h_runtime_top_p;
        shape[0] = batch_size;
        newshape.len = 1;
        float h_runtime_top_p_data[batch_size];
        diopiSize_t h_runtime_top_p_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_runtime_top_p_data)), -1};
        diopiRequireTensor(ctx, &h_runtime_top_p, &newshape, &h_runtime_top_p_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        diopiTensorHandle_t runtime_top_p_forcast = nullptr;
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            char* runtime_top_p_forcast_ptr = reinterpret_cast<char*>(workspace_ptr);
            diopiSize_t runtime_top_p_forcast_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(runtime_top_p_forcast_ptr)), -1};
            diopiRequireTensor(
                ctx, &runtime_top_p_forcast, &newshape, &runtime_top_p_forcast_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_device);
            diopiCastDtype(ctx, runtime_top_p_forcast, runtime_top_p);
            diopiLmdeployCopyD2H(ctx, h_runtime_top_p, runtime_top_p_forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_runtime_top_p, runtime_top_p, false);
        }
        diopiTensorHandle_t h_runtime_initial_top_p;
        float h_runtime_initial_top_p_data[batch_size];
        diopiSize_t h_runtime_initial_top_p_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_runtime_initial_top_p_data)), -1};
        diopiRequireTensor(
            ctx, &h_runtime_initial_top_p, &newshape, &h_runtime_initial_top_p_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiCastDtype(ctx, runtime_top_p_forcast, runtime_initial_top_p);
            diopiLmdeployCopyD2H(ctx, h_runtime_initial_top_p, runtime_top_p_forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_runtime_initial_top_p, runtime_initial_top_p, false);
        }
        diopiTensorHandle_t h_top_p_decay_buf;
        float h_top_p_decay_buf_data[batch_size];
        diopiSize_t h_top_p_decay_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_top_p_decay_buf_data)), -1};
        diopiRequireTensor(ctx, &h_top_p_decay_buf, &newshape, &h_top_p_decay_buf_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiCastDtype(ctx, runtime_top_p_forcast, top_p_decay_buf);
            diopiLmdeployCopyD2H(ctx, h_top_p_decay_buf, runtime_top_p_forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_top_p_decay_buf, top_p_decay_buf, false);
        }
        diopiTensorHandle_t h_top_p_min_buf;
        float h_top_p_min_buf_data[batch_size];
        diopiSize_t h_top_p_min_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_top_p_min_buf_data)), -1};
        diopiRequireTensor(ctx, &h_top_p_min_buf, &newshape, &h_top_p_min_buf_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiCastDtype(ctx, runtime_top_p_forcast, top_p_min_buf);
            diopiLmdeployCopyD2H(ctx, h_top_p_min_buf, runtime_top_p_forcast, false);
        } else {
            diopiLmdeployCopyD2H(ctx, h_top_p_min_buf, top_p_min_buf, false);
        }
        diopiTensorHandle_t h_top_p_reset_ids_buf;
        void* h_top_p_reset_ids_buf_data;
        diopiRequireTensor(ctx, &h_top_p_reset_ids_buf, &newshape, nullptr, intdtype, diopiDevice_t::diopi_host);
        diopiGetTensorData(h_top_p_reset_ids_buf, &h_top_p_reset_ids_buf_data);
        diopiLmdeployCopyD2H(ctx, h_top_p_reset_ids_buf, top_p_reset_ids_buf, false);

        // invokeTopPInitialize
        diopiTensorHandle_t topp_id_vals_buf;
        newshape.len = 2;
        shape[0] = batch_size;
        shape[1] = vocab_size_padded;
        char* topp_id_vals_buf_ptr = reinterpret_cast<char*>(persistent_workspace_ptr);
        diopiSize_t topp_id_vals_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topp_id_vals_buf_ptr)), -1};
        diopiRequireTensor(ctx, &topp_id_vals_buf, &newshape, &topp_id_vals_buf_stride, intdtype, device);
        diopiTensorHandle_t topp_id_vals_buf_forexpand;
        newshape.len = 1;
        shape[0] = batch_size;
        diopiScalar_t toppoffset_start{diopiDtype_t::diopi_dtype_float64, double(0)};
        diopiScalar_t toppoffset_end{diopiDtype_t::diopi_dtype_float64, double(batch_size - 1)};
        char* topp_id_vals_buf_forexpand_ptr = reinterpret_cast<char*>(topp_id_vals_buf_ptr) + intitemsize * batch_size * vocab_size_padded;
        diopiSize_t topp_id_vals_buf_forexpand_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topp_id_vals_buf_forexpand_ptr)), -1};
        diopiRequireTensor(ctx, &topp_id_vals_buf_forexpand, &newshape, &topp_id_vals_buf_forexpand_stride, intdtype, device);
        diopiLinspace(ctx, topp_id_vals_buf_forexpand, &toppoffset_start, &toppoffset_end, batch_size);
        diopiTensorHandle_t topp_id_vals_buf_forexpand_withdim;
        newshape.len = 2;
        shape[0] = batch_size;
        shape[1] = 1;
        diopiRequireTensor(ctx, &topp_id_vals_buf_forexpand_withdim, &newshape, &topp_id_vals_buf_forexpand_stride, intdtype, device);
        diopiExpand(ctx, topp_id_vals_buf, topp_id_vals_buf_forexpand_withdim);
        diopiTensorHandle_t topp_offset_buf;
        newshape.len = 1;
        shape[0] = batch_size + 1;
        char* topp_offset_buf_ptr = reinterpret_cast<char*>(topp_id_vals_buf_ptr) + intitemsize * batch_size * vocab_size_padded;
        diopiSize_t topp_offset_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(topp_offset_buf_ptr)), -1};
        diopiRequireTensor(ctx, &topp_offset_buf, &newshape, &topp_offset_buf_stride, intdtype, device);
        diopiTensorHandle_t begin_topp_offset_buf;
        newshape.len = 1;
        shape[0] = batch_size + 1;
        char* begin_topp_offset_buf_ptr = reinterpret_cast<char*>(topp_offset_buf_ptr) + intitemsize * (batch_size + 1);
        diopiSize_t begin_topp_offset_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(begin_topp_offset_buf_ptr)), -1};
        diopiRequireTensor(ctx, &begin_topp_offset_buf, &newshape, &begin_topp_offset_buf_stride, intdtype, device);
        toppoffset_end.fval = batch_size * vocab_size_padded;
        diopiLinspace(ctx, topp_offset_buf, &toppoffset_start, &toppoffset_end, batch_size + 1);
        diopiLmdeployCopyD2D(ctx, begin_topp_offset_buf, topp_offset_buf, false);

        // addBiasSoftMax. no bias and vocab_size_padded = vocab_size with norm softmax
        double MAX_T_VAL = dtype == diopiDtype_t::diopi_dtype_float16 ? 65504.F : FLT_MAX;
        diopiScalar_t scalar_dmax{dtype, MAX_T_VAL};
        diopiScalar_t scalar_dmin{dtype, -MAX_T_VAL};
        diopiScalar_t scalar_done{dtype, double(1)};
        for (int64_t i = 0; i < batch_size; i++) {
            // no bias and vocab_size_padded = vocab_size
            if (h_finished_data != nullptr && h_finished_data[i]) {
                int64_t end_id = diopiDtype_t::diopi_dtype_int32 == intdtype ? *(reinterpret_cast<int32_t*>(h_end_ids_data) + i)
                                                                             : *(reinterpret_cast<int64_t*>(h_end_ids_data) + i);
                if (end_id < vocab_size_padded) {
                    diopiTensorHandle_t logit_masks;
                    newshape.len = 1;
                    shape[0] = vocab_size_padded;
                    char* logit_masks_ptr = reinterpret_cast<char*>(logits_ptr) + i * itemsize * vocab_size_padded;
                    diopiSize_t logit_masks_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_masks_ptr)), -1};
                    diopiRequireTensor(ctx, &logit_masks, &newshape, &logit_masks_stride, dtype, device);
                    diopiFill(ctx, logit_masks, &scalar_dmin);
                    diopiTensorHandle_t logit_mask;
                    newshape.len = 1;
                    shape[0] = 1;
                    char* logit_mask_ptr = reinterpret_cast<char*>(logits_ptr) + i * itemsize * vocab_size_padded + itemsize * end_id;
                    diopiSize_t logit_mask_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_mask_ptr)), -1};
                    diopiRequireTensor(ctx, &logit_mask, &newshape, &logit_mask_stride, dtype, device);
                    diopiFill(ctx, logit_mask, &scalar_dmax);
                }
            }
        }

        diopiTensorHandle_t logits_forsample;
        newshape.len = 2;
        shape[0] = batch_size;
        shape[1] = vocab_size_padded;
        char* logits_forsample_ptr = reinterpret_cast<char*>(workspace_ptr);
        diopiSize_t logits_forsample_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logits_forsample_ptr)), -1};
        diopiRequireTensor(ctx, &logits_forsample, &newshape, &logits_forsample_stride, dtype, device);
        diopiSoftmax(ctx, logits_forsample, logits, 1);

        diopiScalar_t scalar_dp{dtype, double(0)};
        diopiScalar_t scalar_did{dtype, double(-1)};
        for (int64_t i = 0; i < batch_size; i++) {
            if (h_skip_decode_data != nullptr && h_skip_decode_data[i]) {
                h_runtime_top_p_data[i] = std::max(h_runtime_top_p_data[i] * h_top_p_decay_buf_data[i], h_top_p_min_buf_data[i]);
            }
            float p{h_runtime_top_p_data[i]};  // p
            diopiTensorHandle_t logit;
            newshape.len = 1;
            shape[0] = vocab_size_padded;
            char* logit_ptr = reinterpret_cast<char*>(logits_forsample_ptr) + i * itemsize * vocab_size_padded;
            diopiSize_t logit_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_ptr)), -1};
            diopiRequireTensor(ctx, &logit, &newshape, &logit_stride, dtype, device);
            diopiTensorHandle_t output_id;
            newshape.len = 1;
            shape[0] = 1;
            char* output_id_ptr = reinterpret_cast<char*>(output_ids_ptr) + intitemsize * (step * batch_size + i);
            diopiSize_t output_id_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(output_id_ptr)), -1};
            diopiRequireTensor(ctx, &output_id, &newshape, &output_id_stride, intdtype, device);
            scalar_did.fval = -1;

            diopiTensorHandle_t check;
            newshape.len = 1;
            shape[0] = vocab_size_padded;
            char* check_ptr = reinterpret_cast<char*>(workspace_ptr) + itemsize * batch_size * vocab_size_padded;
            diopiSize_t check_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(check_ptr)), -1};
            diopiRequireTensor(ctx, &check, &newshape, &check_stride, diopiDtype_t::diopi_dtype_bool, device);
            diopiTensorHandle_t h_check;
            bool h_check_data[vocab_size_padded];
            diopiSize_t h_check_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_check_data)), -1};
            diopiRequireTensor(ctx, &h_check, &newshape, &h_check_stride, diopiDtype_t::diopi_dtype_bool, diopiDevice_t::diopi_host);
            scalar_dp.fval = p;
            diopiGeScalar(ctx, check, logit, &scalar_dp);
            diopiLmdeployCopyD2H(ctx, h_check, check, false);
            for (int64_t i = 0; i < vocab_size_padded; i++) {
                if (h_check_data[i]) {
                    scalar_did.fval = i;
                    break;
                }
            }
            if (scalar_did.fval >= 0) {
                diopiFill(ctx, output_id, &scalar_did);
            } else {
                diopiTensorHandle_t rand_num;
                newshape.len = 1;
                shape[0] = 1;
                char* rand_num_ptr = reinterpret_cast<char*>(check_ptr) + sizeof(bool) * vocab_size_padded;
                diopiSize_t rand_num_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(rand_num_ptr)), -1};
                diopiRequireTensor(ctx, &rand_num, &newshape, &rand_num_stride, diopiDtype_t::diopi_dtype_float32, device);
                diopiUniformInp(ctx, rand_num, 0, 1, generators[i]);
                diopiTensorHandle_t h_rand_num;
                float h_rand_num_data[1];
                diopiSize_t h_rand_num_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_rand_num_data)), -1};
                diopiRequireTensor(ctx, &h_rand_num, &newshape, &h_rand_num_stride, diopiDtype_t::diopi_dtype_float32, diopiDevice_t::diopi_host);
                diopiLmdeployCopyD2H(ctx, h_rand_num, rand_num, false);
                float p_threshold{p * h_rand_num_data[0]};  // p_threshold
                p_threshold = 0.5 * p;                      // for test
                diopiTensorHandle_t logit_buf;
                newshape.len = 1;
                shape[0] = vocab_size_padded;
                char* logit_buf_ptr = reinterpret_cast<char*>(check_ptr) + sizeof(bool) * vocab_size_padded;
                diopiSize_t logit_buf_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_buf_ptr)), -1};
                diopiRequireTensor(ctx, &logit_buf, &newshape, &logit_buf_stride, dtype, device);
                diopiTensorHandle_t logit_cumsum;
                char* logit_cumsum_ptr = reinterpret_cast<char*>(logit_buf_ptr) + itemsize * vocab_size_padded;
                diopiSize_t logit_cumsum_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(logit_cumsum_ptr)), -1};
                diopiRequireTensor(ctx, &logit_cumsum, &newshape, &logit_cumsum_stride, dtype, device);
                diopiTensorHandle_t check_index;
                char* check_index_ptr = reinterpret_cast<char*>(logit_cumsum_ptr) + itemsize * vocab_size_padded;
                diopiSize_t check_index_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(check_index_ptr)), -1};
                diopiRequireTensor(ctx, &check_index, &newshape, &check_index_stride, diopiDtype_t::diopi_dtype_int64, device);
                diopiTensorHandle_t h_check_index;
                int64_t h_check_index_data[vocab_size_padded];
                diopiSize_t h_check_index_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(h_check_index_data)), -1};
                diopiRequireTensor(ctx, &h_check_index, &newshape, &h_check_index_stride, diopiDtype_t::diopi_dtype_int64, diopiDevice_t::diopi_host);

                bool stable = true;
                diopiSort(ctx, logit_buf, check_index, logit, 0, true, &stable);
                diopiCumsum(ctx, logit_cumsum, logit_buf, 0);
                scalar_dp.fval = p_threshold;
                diopiGeScalar(ctx, check, logit_cumsum, &scalar_dp);
                diopiLmdeployCopyD2H(ctx, h_check, check, false);
                diopiLmdeployCopyD2H(ctx, h_check_index, check_index, false);
                for (int64_t i = 0; i < vocab_size_padded - 1; i++) {
                    if (h_check_data[i]) {
                        scalar_did.fval = h_check_index_data[i];
                        break;
                    }
                }
                if (scalar_did.fval >= 0) {
                    diopiFill(ctx, output_id, &scalar_did);
                }
            }

            if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                diopiTensorHandle_t top_value;
                newshape.len = 1;
                shape[0] = 1;
                char* top_value_ptr = reinterpret_cast<char*>(logit_ptr) + itemsize * int64_t(scalar_did.fval);
                diopiSize_t top_value_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(top_value_ptr)), -1};
                diopiRequireTensor(ctx, &top_value, &newshape, &top_value_stride, dtype, device);
                diopiLogInp(ctx, top_value);
                if (cum_log_probs != nullptr) {
                    diopiTensorHandle_t cum_log_prob;
                    void* cum_log_probs_ptr;
                    diopiGetTensorData(cum_log_probs, &cum_log_probs_ptr);
                    float* cum_log_prob_ptr = reinterpret_cast<float*>(cum_log_probs_ptr) + i;
                    diopiSize_t cum_log_prob_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(cum_log_prob_ptr)), -1};
                    diopiRequireTensor(ctx, &cum_log_prob, &newshape, &cum_log_prob_stride, diopiDtype_t::diopi_dtype_float32, device);
                    diopiAddInp(ctx, cum_log_prob, top_value, &scalar_done);
                }
                if (output_log_probs != nullptr) {
                    diopiTensorHandle_t output_log_prob;
                    void* output_log_probs_ptr;
                    diopiGetTensorData(output_log_probs, &output_log_probs_ptr);
                    float* output_log_prob_ptr = reinterpret_cast<float*>(output_log_probs_ptr) + i;
                    diopiSize_t output_log_prob_stride{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(output_log_prob_ptr)), -1};
                    diopiRequireTensor(ctx, &output_log_prob, &newshape, &output_log_prob_stride, diopiDtype_t::diopi_dtype_float32, device);
                    if (dtype == diopiDtype_t::diopi_dtype_float32) {
                        diopiLmdeployCopyD2D(ctx, output_log_prob, top_value, false);
                    } else {
                        diopiCastDtype(ctx, output_log_prob, top_value);
                    }
                }
            }
            if (sequence_lengths != nullptr && finished != nullptr) {
                if (!h_finished_data[i]) {
                    if (intdtype == diopiDtype_t::diopi_dtype_int32) {
                        int32_t* h_sequence_length_i = reinterpret_cast<int32_t*>(h_sequence_lengths_data) + i;
                        *h_sequence_length_i += 1;
                    } else {
                        int64_t* h_sequence_length_i = reinterpret_cast<int64_t*>(h_sequence_lengths_data) + i;
                        *h_sequence_length_i += 1;
                    }
                }
                int64_t h_end_ids_i = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(h_end_ids_data) + i)
                                                                                  : *(reinterpret_cast<int64_t*>(h_end_ids_data) + i);
                h_finished_data[i] = int64_t(scalar_did.fval) == h_end_ids_i;
            }
            int64_t h_top_p_reset_ids_i = intdtype == diopiDtype_t::diopi_dtype_int32 ? *(reinterpret_cast<int32_t*>(h_top_p_reset_ids_buf_data) + i)
                                                                                      : *(reinterpret_cast<int64_t*>(h_top_p_reset_ids_buf_data) + i);
            if (int64_t(scalar_did.fval) == h_top_p_reset_ids_i) {
                h_runtime_top_p_data[i] = h_runtime_initial_top_p_data[i];
            } else {
                h_runtime_top_p_data[i] = std::max(h_runtime_top_p_data[i] * h_top_p_decay_buf_data[i], h_top_p_min_buf_data[i]);
            }
        }
        if (sequence_lengths != nullptr && finished != nullptr) {
            diopiLmdeployCopyH2D(ctx, sequence_lengths, h_sequence_lengths, false);
            diopiLmdeployCopyH2D(ctx, finished, h_finished, false);
        }
        if (dtype != diopiDtype_t::diopi_dtype_float32) {
            diopiLmdeployCopyH2D(ctx, runtime_top_p_forcast, h_runtime_top_p, false);
            diopiCastDtype(ctx, runtime_top_p, runtime_top_p_forcast);
        } else {
            diopiLmdeployCopyH2D(ctx, runtime_top_p, h_runtime_top_p, false);
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTransposeAxis01(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiConstTensorHandle_t input, const int64_t dim0,
                                            const int64_t dim1, const int64_t dim2) {
    return diopiSuccess;
};

DIOPI_API diopiError_t diopiPlusScalarInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, const int64_t val, const int64_t size) {
    diopiSize_t in_shape;
    diopiGetTensorShape(inoutput, &in_shape);
    if (in_shape.len != 1) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiDevice_t in_device;
    diopiGetTensorDtype(inoutput, &in_type);
    diopiGetTensorDevice(inoutput, &in_device);

    int64_t front_len = (size <= in_shape.data[0]) ? size : in_shape.data[0];
    diopiSize_t front_shape;
    front_shape.data = &front_len;
    front_shape.len = 1;
    diopiSize_t front_stride;

    void* input_data;
    diopiGetTensorData(inoutput, &input_data);
    front_stride.data = reinterpret_cast<const int64_t*>(input_data);
    front_stride.len = -1;
    diopiTensorHandle_t front;
    diopiRequireTensor(ctx, &front, &front_shape, &front_stride, in_type, in_device);

    diopiScalar_t val_scalar;
    val_scalar.stype = diopi_dtype_int64;
    val_scalar.ival = val;

    diopiScalar_t one;
    one.stype = diopi_dtype_int64;
    one.ival = 1;
    diopiAddInpScalar(ctx, front, &val_scalar, &one);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiUpdatePaddingCount(diopiContextHandle_t ctx, diopiTensorHandle_t total_padding_count, diopiConstTensorHandle_t input_lengths,
                                               int64_t max_input_length, int64_t batch_size) {
    if (input_lengths == nullptr) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;
    diopiGetTensorDtype(input_lengths, &in_type);
    diopiGetTensorShape(input_lengths, &in_shape);
    diopiGetTensorStride(input_lengths, &in_stride);
    diopiGetTensorDevice(input_lengths, &in_device);

    diopiScalar_t max_input_length_scalar;
    max_input_length_scalar.stype = diopi_dtype_float32;
    max_input_length_scalar.fval = double(max_input_length);

    if (total_padding_count == nullptr) {
        diopiRequireTensor(ctx, &total_padding_count, &in_shape, &in_stride, in_type, in_device);
    }

    diopiFill(ctx, total_padding_count, &max_input_length_scalar);

    diopiScalar_t one;
    one.stype = diopi_dtype_int64;
    one.ival = 1;
    diopiSubInp(ctx, total_padding_count, const_cast<diopiConstTensorHandle_t>(input_lengths), &one);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLengthCriterion(diopiContextHandle_t ctx, diopiTensorHandle_t finished, diopiTensorHandle_t should_stop,
                                            diopiTensorHandle_t finished_sum, diopiConstTensorHandle_t sequence_limit_length, int64_t batch_size,
                                            int64_t step) {
    if (finished == nullptr || sequence_limit_length == nullptr) {
        return diopiErrorOccurred;
    }

    diopiScalar_t step_scalar;
    step_scalar.stype = diopi_dtype_float64;
    step_scalar.fval = double(step);

    diopiLeScalar(ctx, finished, sequence_limit_length, &step_scalar);
    diopiTensorHandle_t finished_int32;
    diopiSize_t finished_shape;
    diopiGetTensorShape(finished, &finished_shape);
    diopiRequireTensor(ctx, &finished_int32, &finished_shape, nullptr, diopiDtype_t::diopi_dtype_int32, diopiDevice_t::diopi_device);
    diopiCastDtype(ctx, finished_int32, finished);

    diopiSize_t dim_zero;
    int64_t tmp_zero = 0;
    dim_zero.data = &tmp_zero;
    dim_zero.len = 1;
    diopiSum(ctx, finished_sum, finished_int32, dim_zero);
    diopiAll(ctx, should_stop, finished, &tmp_zero);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t embedding_table,
                                                       diopiConstTensorHandle_t all_ids, const int64_t batch_size, const int64_t hidden_units,
                                                       const int64_t step) {
    if (from_tensor == nullptr || embedding_table == nullptr || all_ids == nullptr) {
        return diopiErrorOccurred;
    }

    diopiDtype_t in_type;
    diopiSize_t in_shape, in_stride;
    diopiDevice_t in_device;

    diopiGetTensorDtype(all_ids, &in_type);
    diopiGetTensorShape(all_ids, &in_shape);
    diopiGetTensorStride(all_ids, &in_stride);
    diopiGetTensorDevice(all_ids, &in_device);

    diopiTensorHandle_t this_step_ids;
    diopiSize_t this_step_ids_shape;
    this_step_ids_shape.len = 1;
    this_step_ids_shape.data = &batch_size;
    diopiRequireTensor(ctx, &this_step_ids, &this_step_ids_shape, nullptr, in_type, in_device);

    sliceAsSelect(ctx, this_step_ids, all_ids, 0, step);
    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, this_step_ids);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiInputIdsEmbeddingLookupPosEncoding(diopiContextHandle_t ctx, diopiTensorHandle_t from_tensor, diopiConstTensorHandle_t input_ids,
                                                               diopiConstTensorHandle_t embedding_table, const int64_t input_lengths,
                                                               const int64_t hidden_units) {
    if (from_tensor == nullptr || input_ids == nullptr || embedding_table == nullptr) {
        return diopiErrorOccurred;
    }

    diopiIndexSelect(ctx, from_tensor, embedding_table, 0, input_ids);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiStopWordsCriterion(diopiContextHandle_t ctx, diopiConstTensorHandle_t output_ids, diopiConstTensorHandle_t stop_words,
                                               diopiTensorHandle_t finished, int64_t id_offset, int64_t stop_words_len, int64_t batch_size, int64_t step) {
    // always id_offset = 0
    if (output_ids == nullptr || stop_words == nullptr || finished == nullptr) {
        return diopiErrorOccurred;
    }

    diopiTensorHandle_t stop_words_host;
    diopiSize_t stop_words_shape;
    diopiDtype_t stop_words_type;
    diopiGetTensorDtype(stop_words, &stop_words_type);
    if (stop_words_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    diopiGetTensorShape(stop_words, &stop_words_shape);
    diopiRequireTensor(ctx, &stop_words_host, &stop_words_shape, nullptr, diopi_dtype_int32, diopi_host);
    diopiLmdeployCopyD2H(ctx, stop_words_host, stop_words, false);

    const int32_t* stop_words_ptr;
    int32_t* stop_words_host_ptr;

    int64_t finished_elem_size;
    diopiGetTensorElemSize(finished, &finished_elem_size);
    diopiDtype_t finished_type;
    diopiGetTensorDtype(finished, &finished_type);
    assert(finished_type == diopi_dtype_bool);

    diopiGetTensorDataConst(stop_words, reinterpret_cast<const void**>(&stop_words_ptr));
    diopiGetTensorData(stop_words_host, reinterpret_cast<void**>(&stop_words_host_ptr));

    diopiDtype_t ids_type;
    diopiGetTensorDtype(output_ids, &ids_type);
    if (ids_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t* base_stop_words_host = stop_words_host_ptr + batch_idx * 2 * stop_words_len;
        const int32_t* base_offsets_host = base_stop_words_host + stop_words_len;
        const int32_t* base_stop_word = stop_words_ptr + batch_idx * 2 * stop_words_len;

        for (int64_t stop_word_idx = 0; stop_word_idx < stop_words_len; ++stop_word_idx) {
            if (base_offsets_host[stop_word_idx] < 0) {
                continue;
            }
            const int32_t stop_word_start_idx = (stop_word_idx > 0) ? base_offsets_host[stop_word_idx - 1] : 0;
            const int32_t stop_word_end_idx = base_offsets_host[stop_word_idx] - 1;
            const int64_t stop_word_len = stop_word_end_idx - stop_word_start_idx + 1;

            if (step + 1 < stop_word_len) {
                continue;
            }

            diopiTensorHandle_t stop_word_tensor;
            diopiSize_t stop_word_shape;
            stop_word_shape.len = 1;
            stop_word_shape.data = &stop_word_len;
            diopiDevice_t stop_word_device;
            diopiGetTensorDevice(stop_words, &stop_word_device);
            diopiSize_t stride;
            stride.len = -1;
            stride.data = reinterpret_cast<const int64_t*>(base_stop_word + stop_word_start_idx);
            diopiRequireTensor(ctx, &stop_word_tensor, &stop_word_shape, &stride, ids_type, stop_word_device);

            diopiTensorHandle_t output_ids_col;
            diopiGetTensorStride(stop_word_tensor, &stride);
            diopiSize_t output_ids_col_shape;
            output_ids_col_shape.len = 1;
            int64_t output_ids_col_shape_tmp = step + 1;
            output_ids_col_shape.data = &output_ids_col_shape_tmp;
            diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, ids_type, diopi_device);
            sliceAsSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

            diopiTensorHandle_t output_ids_to_compare;
            char* output_ids_to_compare_data;
            diopiGetTensorData(output_ids_col, reinterpret_cast<void**>(&output_ids_to_compare_data));
            int64_t elem_size;
            diopiGetTensorElemSize(output_ids_col, &elem_size);
            output_ids_to_compare_data += (step - stop_word_len + 1) * elem_size;
            stride.len = -1;
            stride.data = reinterpret_cast<const int64_t*>(reinterpret_cast<int64_t*>(output_ids_to_compare_data));
            diopiRequireTensor(ctx, &output_ids_to_compare, &stop_word_shape, &stride, ids_type, diopi_device);

            diopiTensorHandle_t cmp_res;
            diopiRequireTensor(ctx, &cmp_res, &stop_word_shape, nullptr, diopi_dtype_bool, stop_word_device);
            diopiEq(ctx, cmp_res, output_ids_to_compare, stop_word_tensor);

            diopiTensorHandle_t cmp_res_sum;
            diopiSize_t cmp_res_sum_shape;
            cmp_res_sum_shape.len = 1;
            int64_t tmp_one = 1;
            cmp_res_sum_shape.data = &tmp_one;
            diopiRequireTensor(ctx, &cmp_res_sum, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, stop_word_device);
            int64_t tmp_zero = 0;
            diopiAll(ctx, cmp_res_sum, cmp_res, &tmp_zero);

            diopiTensorHandle_t cmp_res_sum_host;
            diopiRequireTensor(ctx, &cmp_res_sum_host, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, diopi_host);
            diopiLmdeployCopyD2H(ctx, cmp_res_sum_host, cmp_res_sum, false);
            bool* cmp_res_sum_host_data;
            diopiGetTensorData(cmp_res_sum_host, reinterpret_cast<void**>(&cmp_res_sum_host_data));

            if (cmp_res_sum_host_data[0]) {
                diopiScalar_t true_scalar;
                true_scalar.stype = diopi_dtype_bool;
                true_scalar.ival = 1;

                diopiTensorHandle_t finished_to_modify;
                diopiSize_t finished_to_modify_stride;
                finished_to_modify_stride.len = -1;
                finished_to_modify_stride.data = getDataOffsetPtr(finished, batch_idx);

                diopiRequireTensor(ctx, &finished_to_modify, &cmp_res_sum_shape, &finished_to_modify_stride, diopi_dtype_bool, diopi_device);

                diopiFill(ctx, finished_to_modify, &true_scalar);

                break;
            }
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBanBadWordsInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t output_ids,
                                           diopiConstTensorHandle_t bad_words, int64_t id_offset, int64_t bad_words_len, bool share_words, int64_t batch_size,
                                           int64_t vocab_size, int64_t step) {
    // always id_offset = 0
    if (logits == nullptr || output_ids == nullptr || bad_words == nullptr) {
        return diopiErrorOccurred;
    }

    diopiTensorHandle_t bad_words_host;
    diopiSize_t bad_words_shape, bad_words_stride;
    diopiGetTensorShape(bad_words, &bad_words_shape);
    diopiGetTensorStride(bad_words, &bad_words_stride);
    diopiRequireTensor(ctx, &bad_words_host, &bad_words_shape, &bad_words_stride, diopi_dtype_int32, diopi_host);
    diopiLmdeployCopyD2H(ctx, bad_words_host, bad_words, false);

    const int32_t* bad_words_ptr;
    int32_t* bad_words_host_ptr;

    diopiGetTensorDataConst(bad_words, reinterpret_cast<const void**>(&bad_words_ptr));
    diopiGetTensorData(bad_words_host, reinterpret_cast<void**>(&bad_words_host_ptr));

    diopiDtype_t ids_type;
    diopiGetTensorDtype(output_ids, &ids_type);
    if (ids_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t* base_bad_words_host = share_words ? bad_words_host_ptr : bad_words_host_ptr + batch_idx * 2 * bad_words_len;
        const int32_t* base_offsets_host = base_bad_words_host + bad_words_len;
        const int32_t* base_bad_word = share_words ? bad_words_ptr : bad_words_ptr + batch_idx * 2 * bad_words_len;

        for (int64_t bad_word_idx = 0; bad_word_idx < bad_words_len; ++bad_word_idx) {
            if (base_offsets_host[bad_word_idx] < 0) {
                continue;
            }
            const int32_t bad_word_start_idx = (bad_word_idx > 0) ? base_offsets_host[bad_word_idx - 1] : 0;
            const int32_t bad_word_end_idx = base_offsets_host[bad_word_idx];
            const int64_t bad_word_len = bad_word_end_idx - bad_word_start_idx;

            if (step + 1 < bad_word_len || bad_word_len < 1) {
                continue;
            }

            bool* cmp_res_sum_host_data = nullptr;
            if (bad_word_len != 1) {
                diopiTensorHandle_t bad_word_tensor;
                diopiSize_t bad_word_shape;
                bad_word_shape.len = 1;
                int64_t bad_word_to_compare_len = bad_word_len - 1;
                bad_word_shape.data = &bad_word_to_compare_len;
                diopiDevice_t bad_word_device;
                diopiGetTensorDevice(bad_words, &bad_word_device);
                diopiSize_t stride;
                stride.len = -1;
                stride.data = reinterpret_cast<const int64_t*>(base_bad_word + bad_word_start_idx);
                diopiRequireTensor(ctx, &bad_word_tensor, &bad_word_shape, &stride, ids_type, bad_word_device);

                diopiTensorHandle_t output_ids_col;
                diopiGetTensorStride(bad_word_tensor, &stride);
                diopiSize_t output_ids_col_shape;
                output_ids_col_shape.len = 1;
                output_ids_col_shape.data = &step;
                diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, ids_type, diopi_device);
                sliceAsSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

                char* output_ids_col_data;
                diopiGetTensorData(output_ids_col, reinterpret_cast<void**>(&output_ids_col_data));
                int64_t elem_size;
                diopiGetTensorElemSize(output_ids_col, &elem_size);
                output_ids_col_data += (step - bad_word_len + 1) * elem_size;

                stride.len = -1;
                stride.data = reinterpret_cast<const int64_t*>(reinterpret_cast<int64_t*>(output_ids_col_data));
                diopiTensorHandle_t output_ids_to_compare;
                diopiRequireTensor(ctx, &output_ids_to_compare, &bad_word_shape, &stride, ids_type, diopi_device);

                diopiTensorHandle_t cmp_res;
                diopiRequireTensor(ctx, &cmp_res, &bad_word_shape, nullptr, diopi_dtype_bool, bad_word_device);
                diopiEq(ctx, cmp_res, output_ids_to_compare, bad_word_tensor);

                diopiTensorHandle_t cmp_res_sum;
                diopiSize_t cmp_res_sum_shape;
                cmp_res_sum_shape.len = 1;
                int64_t tmp_one = 1;
                cmp_res_sum_shape.data = &tmp_one;
                diopiRequireTensor(ctx, &cmp_res_sum, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, bad_word_device);
                int64_t tmp_zero = 0;
                diopiAll(ctx, cmp_res_sum, cmp_res, &tmp_zero);

                diopiTensorHandle_t cmp_res_sum_host;
                diopiRequireTensor(ctx, &cmp_res_sum_host, &cmp_res_sum_shape, nullptr, diopi_dtype_bool, diopi_host);
                diopiLmdeployCopyD2H(ctx, cmp_res_sum_host, cmp_res_sum, false);

                diopiGetTensorData(cmp_res_sum_host, reinterpret_cast<void**>(&cmp_res_sum_host_data));
            }

            if (bad_word_len == 1 || (cmp_res_sum_host_data != nullptr && cmp_res_sum_host_data[0])) {
                int32_t banned_token = base_bad_words_host[bad_word_end_idx - 1];
                if (0 < banned_token && banned_token < vocab_size) {
                    int64_t tmp_one = 1;
                    diopiTensorHandle_t logit_to_modify;
                    diopiDtype_t logits_type;
                    diopiGetTensorDtype(logits, &logits_type);
                    diopiSize_t logit_to_modify_shape, logit_to_modify_stride;
                    logit_to_modify_shape.len = 1;
                    logit_to_modify_shape.data = &tmp_one;
                    logit_to_modify_stride.len = -1;
                    char* logit_to_modify_data;
                    diopiGetTensorData(logits, reinterpret_cast<void**>(&logit_to_modify_data));
                    int64_t elem_size;
                    diopiGetTensorElemSize(logits, &elem_size);
                    logit_to_modify_data += (batch_idx * vocab_size + banned_token) * elem_size;
                    logit_to_modify_stride.data = reinterpret_cast<const int64_t*>(logit_to_modify_data);
                    diopiRequireTensor(ctx, &logit_to_modify, &logit_to_modify_shape, &logit_to_modify_stride, logits_type, diopi_device);

                    diopiScalar_t minus_inf;
                    minus_inf.stype = logits_type;
                    minus_inf.fval = -INFINITY;
                    diopiFill(ctx, logit_to_modify, &minus_inf);
                }
                continue;
            }
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGatherOutput(diopiContextHandle_t ctx, diopiTensorHandle_t output_ids, diopiConstTensorHandle_t ids,
                                         diopiConstTensorHandle_t context_length, int64_t max_context_len, int64_t max_gen_step, int64_t max_output_len,
                                         int64_t batch_size) {
    diopiDtype_t ids_type, context_length_type;
    diopiGetTensorDtype(ids, &ids_type);
    diopiGetTensorDtype(context_length, &context_length_type);
    if (context_length_type != diopi_dtype_int32) {
        return diopiErrorOccurred;
    }

    int64_t ids_elem_size;
    diopiGetTensorElemSize(ids, &ids_elem_size);

    diopiTensorHandle_t context_length_host;
    diopiSize_t context_length_shape;
    diopiGetTensorShape(context_length, &context_length_shape);
    diopiRequireTensor(ctx, &context_length_host, &context_length_shape, nullptr, context_length_type, diopi_host);
    diopiLmdeployCopyD2H(ctx, context_length_host, context_length, false);

    int32_t* context_length_host_data;
    diopiGetTensorData(context_length_host, reinterpret_cast<void**>(&context_length_host_data));

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        diopiTensorHandle_t src_col;
        diopiSize_t src_col_shape;
        src_col_shape.len = 1;
        src_col_shape.data = &max_output_len;

        diopiRequireTensor(ctx, &src_col, &src_col_shape, nullptr, ids_type, diopi_device);
        sliceAsSelect(ctx, src_col, ids, 1, batch_idx);

        diopiTensorHandle_t src_col_front;
        diopiSize_t src_col_front_shape, src_col_front_stride;
        int64_t context_len = static_cast<int64_t>(context_length_host_data[batch_idx]);
        src_col_front_shape.len = 1;
        src_col_front_shape.data = &context_len;
        src_col_front_stride.len = -1;
        char* src_col_front_data;
        diopiGetTensorData(src_col, reinterpret_cast<void**>(&src_col_front_data));
        src_col_front_stride.data = reinterpret_cast<const int64_t*>(src_col_front_data);
        diopiRequireTensor(ctx, &src_col_front, &src_col_front_shape, &src_col_front_stride, ids_type, diopi_device);

        diopiTensorHandle_t dst_row_front;
        diopiSize_t dst_row_front_stride;
        dst_row_front_stride.len = -1;
        char* dst_row_front_data;
        diopiGetTensorData(output_ids, reinterpret_cast<void**>(&dst_row_front_data));
        dst_row_front_data += (batch_idx * max_output_len * ids_elem_size);
        dst_row_front_stride.data = reinterpret_cast<const int64_t*>(dst_row_front_data);
        diopiRequireTensor(ctx, &dst_row_front, &src_col_front_shape, &dst_row_front_stride, ids_type, diopi_device);

        diopiLmdeployCopyD2D(ctx, dst_row_front, src_col_front, false);
        if (max_context_len < max_gen_step) {
            diopiTensorHandle_t src_col_back;
            diopiSize_t src_col_back_shape, src_col_back_stride;
            src_col_back_shape.len = 1;
            int64_t back_len = max_gen_step - max_context_len;
            src_col_back_shape.data = &back_len;
            src_col_back_stride.len = -1;
            char* src_col_back_data;
            diopiGetTensorData(src_col, reinterpret_cast<void**>(&src_col_back_data));
            src_col_back_data += (max_context_len * ids_elem_size);
            src_col_back_stride.data = reinterpret_cast<const int64_t*>(src_col_back_data);
            diopiRequireTensor(ctx, &src_col_back, &src_col_back_shape, &src_col_back_stride, ids_type, diopi_device);
            diopiTensorHandle_t dst_row_back;
            diopiSize_t dst_row_back_stride;
            dst_row_back_stride.len = -1;
            char* dst_row_back_data;
            diopiGetTensorData(output_ids, reinterpret_cast<void**>(&dst_row_back_data));
            dst_row_back_data += ((batch_idx * max_output_len + context_len) * ids_elem_size);
            dst_row_back_stride.data = reinterpret_cast<const int64_t*>(dst_row_back_data);
            diopiRequireTensor(ctx, &dst_row_back, &src_col_back_shape, &dst_row_back_stride, ids_type, diopi_device);
            diopiLmdeployCopyD2D(ctx, dst_row_back, src_col_back, false);
        }
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchApplyRepetitionPenaltyInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t penalties,
                                                           diopiConstTensorHandle_t output_ids, const int64_t batch_size, const int64_t vocab_size,
                                                           diopiConstTensorHandle_t input_lengths, const int64_t max_input_length, const int64_t step,
                                                           const int64_t penalty_type) {
    if (logits == nullptr || penalties == nullptr || output_ids == nullptr) {
        return diopiErrorOccurred;
    }

    if (penalty_type != 0 && penalty_type != 1 && penalty_type != 2) {
        return diopiErrorOccurred;
    }

    if (penalty_type == 0) {
        return diopiSuccess;
    }

    diopiDtype_t logits_type, penalties_type, input_lengths_type, output_ids_type;
    diopiGetTensorDtype(logits, &logits_type);
    diopiGetTensorDtype(penalties, &penalties_type);
    diopiGetTensorDtype(output_ids, &output_ids_type);

    int64_t logits_elem_size, penalties_elem_size, input_lengths_elem_size, output_ids_elem_size;
    diopiGetTensorElemSize(logits, &logits_elem_size);
    diopiGetTensorElemSize(penalties, &penalties_elem_size);
    diopiGetTensorElemSize(output_ids, &output_ids_elem_size);

    int32_t* input_lengths_host_data = nullptr;
    if (input_lengths != nullptr) {
        diopiGetTensorElemSize(input_lengths, &input_lengths_elem_size);
        diopiGetTensorDtype(input_lengths, &input_lengths_type);
        if (input_lengths_type != diopi_dtype_int32) {
            return diopiErrorOccurred;
        }
        diopiTensorHandle_t input_lengths_host;
        diopiSize_t input_lengths_shape;
        diopiGetTensorShape(input_lengths, &input_lengths_shape);
        diopiRequireTensor(ctx, &input_lengths_host, &input_lengths_shape, nullptr, input_lengths_type, diopi_host);

        diopiLmdeployCopyD2H(ctx, input_lengths_host, input_lengths, false);
        diopiGetTensorData(input_lengths_host, reinterpret_cast<void**>(&input_lengths_host_data));
    }

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        diopiTensorHandle_t output_ids_col;
        diopiSize_t output_ids_col_shape;
        output_ids_col_shape.len = 1;
        output_ids_col_shape.data = &step;
        diopiRequireTensor(ctx, &output_ids_col, &output_ids_col_shape, nullptr, output_ids_type, diopi_device);

        sliceAsSelect(ctx, output_ids_col, output_ids, 1, batch_idx);

        diopiTensorHandle_t output_ids_col_front;
        diopiSize_t output_ids_col_front_shape, output_ids_col_front_stride;

        output_ids_col_front_shape.len = 1;
        int64_t input_len = input_lengths_host_data == nullptr ? max_input_length : static_cast<int64_t>(input_lengths_host_data[batch_idx]);
        output_ids_col_front_shape.data = &input_len;

        output_ids_col_front_stride.len = -1;
        char* output_ids_col_front_data;
        diopiGetTensorData(output_ids_col, reinterpret_cast<void**>(&output_ids_col_front_data));
        output_ids_col_front_stride.data = reinterpret_cast<const int64_t*>(output_ids_col_front_data);

        diopiRequireTensor(ctx, &output_ids_col_front, &output_ids_col_front_shape, &output_ids_col_front_stride, output_ids_type, diopi_device);
        diopiTensorHandle_t valid_output_ids_col = output_ids_col_front;
        int64_t vaild_output_ids_col_len = input_len;

        if (max_input_length < step) {
            diopiTensorHandle_t output_ids_col_back;
            diopiSize_t output_ids_col_back_shape, output_ids_col_back_stride;
            output_ids_col_back_shape.len = 1;
            int64_t back_len = step - max_input_length;
            output_ids_col_back_shape.data = &back_len;
            output_ids_col_back_stride.len = -1;
            char* output_ids_col_back_data;
            diopiGetTensorData(output_ids_col, reinterpret_cast<void**>(&output_ids_col_back_data));
            output_ids_col_back_data += (max_input_length * output_ids_elem_size);
            output_ids_col_back_stride.data = reinterpret_cast<const int64_t*>(output_ids_col_back_data);
            diopiRequireTensor(ctx, &output_ids_col_back, &output_ids_col_back_shape, &output_ids_col_back_stride, output_ids_type, diopi_device);

            vaild_output_ids_col_len = input_len + back_len;
            diopiSize_t valid_output_ids_col_shape;
            valid_output_ids_col_shape.len = 1;
            valid_output_ids_col_shape.data = &vaild_output_ids_col_len;
            diopiRequireTensor(ctx, &valid_output_ids_col, &valid_output_ids_col_shape, nullptr, output_ids_type, diopi_device);
            diopiConstTensorHandle_t to_cat[2] = {output_ids_col_front, output_ids_col_back};
            combAsCat(ctx, valid_output_ids_col, to_cat, 2, 0);
        }

        diopiTensorHandle_t logits_this_batch;
        diopiSize_t logits_this_batch_shape, logits_this_batch_stride;
        logits_this_batch_shape.len = 1;
        logits_this_batch_shape.data = &vocab_size;
        logits_this_batch_stride.len = -1;
        char* logits_this_batch_data;
        diopiGetTensorData(logits, reinterpret_cast<void**>(&logits_this_batch_data));
        logits_this_batch_data += (batch_idx * vocab_size * logits_elem_size);
        logits_this_batch_stride.data = reinterpret_cast<const int64_t*>(logits_this_batch_data);
        diopiRequireTensor(ctx, &logits_this_batch, &logits_this_batch_shape, &logits_this_batch_stride, logits_type, diopi_device);

        diopiTensorHandle_t logits_to_penalize;
        diopiSize_t logits_to_penalize_shape;
        logits_to_penalize_shape.len = 1;
        logits_to_penalize_shape.data = &vaild_output_ids_col_len;
        diopiRequireTensor(ctx, &logits_to_penalize, &logits_to_penalize_shape, nullptr, logits_type, diopi_device);
        diopiIndexSelect(ctx, logits_to_penalize, logits_this_batch, 0, valid_output_ids_col);

        diopiTensorHandle_t penalties_this_batch;
        diopiSize_t penalties_this_batch_shape, penalties_this_batch_stride;
        penalties_this_batch_shape.len = 1;
        int64_t tmp_one = 1;
        penalties_this_batch_shape.data = &tmp_one;

        penalties_this_batch_stride.len = -1;
        const char* penalties_this_batch_data;
        diopiGetTensorDataConst(penalties, reinterpret_cast<const void**>(&penalties_this_batch_data));
        penalties_this_batch_data += (batch_idx * penalties_elem_size);
        penalties_this_batch_stride.data = reinterpret_cast<const int64_t*>(penalties_this_batch_data);
        diopiRequireTensor(ctx, &penalties_this_batch, &penalties_this_batch_shape, &penalties_this_batch_stride, penalties_type, diopi_device);

        diopiTensorHandle_t penalties_this_batch_expand;
        diopiSize_t penalties_this_batch_expand_shape;
        penalties_this_batch_expand_shape.len = 1;
        penalties_this_batch_expand_shape.data = &vaild_output_ids_col_len;
        diopiRequireTensor(ctx, &penalties_this_batch_expand, &penalties_this_batch_expand_shape, nullptr, penalties_type, diopi_device);
        diopiExpand(ctx, penalties_this_batch_expand, penalties_this_batch);
        diopiScalar_t one_scalar;
        one_scalar.stype = diopi_dtype_int64;
        one_scalar.ival = 1;

        if (penalty_type == 1) {
            diopiSubInp(ctx, logits_to_penalize, penalties_this_batch_expand, &one_scalar);
        } else {
            diopiTensorHandle_t ge_zero_mask, lt_zero_mask;
            diopiSize_t ge_than_zero_mask_shape;
            ge_than_zero_mask_shape.len = 1;
            ge_than_zero_mask_shape.data = &vaild_output_ids_col_len;
            diopiRequireTensor(ctx, &ge_zero_mask, &ge_than_zero_mask_shape, nullptr, diopi_dtype_bool, diopi_device);
            diopiRequireTensor(ctx, &lt_zero_mask, &ge_than_zero_mask_shape, nullptr, diopi_dtype_bool, diopi_device);

            diopiScalar_t zero_scalar;
            zero_scalar.stype = diopi_dtype_float32;
            zero_scalar.ival = 0.0f;
            diopiTensorHandle_t mul_penalty, div_penalty;
            diopiRequireTensor(ctx, &mul_penalty, &logits_to_penalize_shape, nullptr, logits_type, diopi_device);
            diopiRequireTensor(ctx, &div_penalty, &logits_to_penalize_shape, nullptr, logits_type, diopi_device);

            diopiClampScalar(ctx, mul_penalty, logits_to_penalize, nullptr, &zero_scalar);
            diopiClampScalar(ctx, div_penalty, logits_to_penalize, &zero_scalar, nullptr);
            diopiMulInp(ctx, mul_penalty, penalties_this_batch_expand);
            diopiDivInp(ctx, div_penalty, penalties_this_batch_expand, RoundModeNone);
            diopiAdd(ctx, logits_to_penalize, mul_penalty, div_penalty, &one_scalar);
        }
        char reduce[0];
        diopiTensorHandle_t scatter_index = valid_output_ids_col;

        if (output_ids_type != diopi_dtype_int64) {
            diopiSize_t scatter_index_shape;
            diopiGetTensorShape(valid_output_ids_col, &scatter_index_shape);
            diopiRequireTensor(ctx, &scatter_index, &scatter_index_shape, nullptr, diopi_dtype_int64, diopi_device);
            diopiCastDtype(ctx, scatter_index, valid_output_ids_col);
        }

        diopiScatterInp(ctx, logits_this_batch, 0, logits_to_penalize, scatter_index, reduce);
    }

    return diopiSuccess;
}

diopiError_t diopiBatchApplyTemperaturePenaltyInp(diopiContextHandle_t ctx, diopiTensorHandle_t logits, diopiConstTensorHandle_t bias,
                                                  diopiConstTensorHandle_t temperatures, const int64_t batch_size, const int64_t vocab_size,
                                                  const int64_t vocab_size_padd) {
    assert(vocab_size_padd >= vocab_size);
    assert(logits != nullptr);
    assert(temperatures != nullptr);
    diopiDtype_t logits_dtype;
    diopiGetTensorDtype(logits, &logits_dtype);
    diopiSize_t logits_shape;
    diopiGetTensorShape(logits, &logits_shape);
    assert(logits_shape.len == 2 && logits_shape.data[0] == batch_size && logits_shape.data[1] == vocab_size_padd);
    diopiTensorHandle_t lhs;
    std::vector<int64_t> lhs_shape_vec = {batch_size, vocab_size};
    diopiSize_t lhs_shape{lhs_shape_vec.data(), 2};
    diopiRequireTensor(ctx, &lhs, &lhs_shape, nullptr, logits_dtype, diopi_device);
    diopiSlice(ctx, lhs, logits, 1, 0, vocab_size, 1);

    diopiTensorHandle_t rhs = nullptr;
    if (vocab_size_padd > vocab_size) {
        std::vector<int64_t> rhs_shape_vec = {batch_size, vocab_size_padd - vocab_size};
        diopiSize_t rhs_shape{rhs_shape_vec.data(), 2};
        diopiRequireTensor(ctx, &rhs, &rhs_shape, nullptr, logits_dtype, diopi_device);
        diopiSlice(ctx, rhs, logits, 1, vocab_size, vocab_size_padd, 1);
        double MAX_T_VAL = (logits_dtype == diopiDtype_t::diopi_dtype_float16 ? 65504.F : FLT_MAX);
        diopiScalar_t scalar_val;
        scalar_val.stype = logits_dtype;
        scalar_val.fval = -MAX_T_VAL;
        diopiFill(ctx, rhs, &scalar_val);
    }
    diopiDtype_t temperatures_dtype;
    diopiGetTensorDtype(temperatures, &temperatures_dtype);
    diopiTensorHandle_t new_temperatures = nullptr;
    diopiSize_t temperatures_shape;
    diopiGetTensorShape(temperatures, &temperatures_shape);
    diopiRequireTensor(ctx, &new_temperatures, &temperatures_shape, nullptr, temperatures_dtype, diopi_device);

    assert(temperatures_dtype == diopi_dtype_float32);

    diopiScalar_t eps_scalar;
    eps_scalar.stype = temperatures_dtype;
    eps_scalar.fval = 1e-6;
    diopiScalar_t one_scalar;
    one_scalar.stype = temperatures_dtype;
    one_scalar.fval = 1.0;
    diopiAddScalar(ctx, new_temperatures, temperatures, &eps_scalar, &one_scalar);

    if (bias != nullptr) {
        diopiScalar_t t;
        t.stype = logits_dtype;
        t.fval = 1.0;
        diopiAddInp(ctx, lhs, bias, &t);
    }

    diopiSize_t new_temperatures_shape;
    diopiGetTensorShape(new_temperatures, &new_temperatures_shape);
    diopiTensorHandle_t new_temperatures_host;
    diopiRequireTensor(ctx, &new_temperatures_host, &new_temperatures_shape, nullptr, temperatures_dtype, diopi_host);
    diopiLmdeployCopyD2H(ctx, new_temperatures_host, new_temperatures, false);
    char* new_temperatures_host_data;

    diopiGetTensorData(new_temperatures_host, reinterpret_cast<void**>(&new_temperatures_host_data));
    for (int64_t i = 0; i < batch_size; ++i) {
        diopiScalar_t temperature_scalar;
        temperature_scalar.stype = diopi_dtype_float32;
        temperature_scalar.fval = reinterpret_cast<float*>(new_temperatures_host_data)[i];

        diopiTensorHandle_t logits_row;
        diopiSize_t logits_row_shape, logits_row_stride;
        logits_row_shape.len = 1;
        logits_row_shape.data = &vocab_size;

        logits_row_stride.len = -1;
        logits_row_stride.data = getDataOffsetPtr(lhs, i * vocab_size);
        diopiRequireTensor(ctx, &logits_row, &logits_row_shape, &logits_row_stride, logits_dtype, diopi_device);
        diopiDivInpScalar(ctx, logits_row, &temperature_scalar, RoundModeNone);
    }

    if (rhs == nullptr) {
        diopiCopyInp(ctx, lhs, logits);
    } else {
        std::array<diopiConstTensorHandle_t, 2> tensors = {lhs, rhs};
        combAsCat(ctx, logits, tensors.data(), tensors.size(), 1);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
