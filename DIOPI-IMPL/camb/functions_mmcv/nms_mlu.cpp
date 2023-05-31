/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <memory>

#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void KernelNms(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, const cnrtDataType_t data_type_input, const void *boxes_ptr,
               const void *scores_ptr, const int input_num_boxes, const int max_output_boxes, const float iou_threshold, const float offset,
               void *workspace_ptr, void *output_size_ptr, void *output_ptr);
}  // namespace camb

}  // namespace impl

int selectUnionType(uint32_t use_job, int box_num_per_core) {
    // the box_num_per_core should be at least 256, otherwise the real IO
    // bandwidth would be very low
    while (box_num_per_core < 256 && use_job >= 4) {
        box_num_per_core *= 2;
        use_job /= 2;
    }
    return use_job;
}

static cnnlStatus_t policyFunc(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type, int &core_num_per_class, const int input_box_num) {
    uint32_t core_dim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    uint32_t cluster_number = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    uint32_t job_limit = impl::camb::getJobLimitCapability();
    uint32_t core_number = job_limit;

    int box_num_per_core = (input_box_num + core_number - 1) / core_number;
    int use_job = selectUnionType(job_limit, box_num_per_core);
    // initiate k_type as Union1
    k_dim->x = core_dim;
    k_dim->y = 1;
    k_dim->z = 1;
    *k_type = CNRT_FUNC_TYPE_UNION1;
    switch (job_limit) {
        case CN_KERNEL_CLASS_BLOCK:
        case CN_KERNEL_CLASS_UNION:
        case CN_KERNEL_CLASS_UNION2:
        case CN_KERNEL_CLASS_UNION4:
        case CN_KERNEL_CLASS_UNION8:
        case CN_KERNEL_CLASS_UNION16: {
            if (use_job < 4) {
                k_dim->x = 1;
                *k_type = CNRT_FUNC_TYPE_BLOCK;
            } else if (use_job == 4) {
                k_dim->x = core_dim;
                *k_type = CNRT_FUNC_TYPE_UNION1;
            } else {
                k_dim->x = use_job;
                *k_type = (cnrtFunctionType_t)use_job;
            }
        }; break;
        default:
            DIOPI_CHECK_ABORT(false,
                              "%s",
                              "[cnnlNms_v2]: got unsupported job limit number. Use "
                              "default CN_KERNEL_CLASS_UNION1 with UNION1 task.");
    }
    return CNNL_STATUS_SUCCESS;
}

extern "C" DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out_, diopiConstTensorHandle_t boxes_,
                                               diopiConstTensorHandle_t scores_, double iou_threshold, int64_t offset) {
    auto boxes = impl::camb::DiopiTensor(boxes_);
    auto scores = impl::camb::DiopiTensor(scores_);

    if (boxes.numel() == 0) {
        diopiScalar_t scalar = {diopi_dtype_int64, 1};
        auto temp_coors = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int64);
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(temp_coors), &scalar));
        *out_ = diopiTensorHandle_t(temp_coors);
        return diopiSuccess;
    }

    int input_num_boxes = boxes.size(0);
    int max_output_boxes = boxes.size(0);

    cnrtDataType_t data_type_input = impl::camb::dtype2CnrtDtype(boxes.dtype());
    cnrtDim3_t k_dim;
    cnrtJobType_t k_type;

    int core_num_per_class;
    policyFunc(&k_dim, &k_type, core_num_per_class, input_num_boxes);

    // transpose boxes (n, 4) to (4, n) for better performance
    auto boxes_t = impl::camb::requiresTensor(ctx, {boxes.size(1), boxes.size(0)}, boxes.dtype());

    DIOPI_CALL(diopiTranspose(ctx, diopiTensorHandle_t(boxes_t), diopiTensorHandle_t(boxes), 0, 1));
    auto output = impl::camb::requiresTensor(ctx, {max_output_boxes}, diopi_dtype_int32);
    auto output_size = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int32);

    // workspace
    const int info_num = 5;  // x1, x2, y1, y2 and score
    size_t space_size = 0;
    if (boxes.dtype() == diopi_dtype_float16) {
        space_size = input_num_boxes * sizeof(int16_t) * info_num + sizeof(float);
    } else {
        space_size = input_num_boxes * sizeof(float) * info_num + sizeof(float);
    }
#if __BANG_ARCH__ > 370
    int cluster_num = getCoreNumOfJobLimitCapability() / impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    space_size += cluster_number * sizeof(float) * 7;
#endif
    auto workspace = impl::camb::requiresTensor(ctx, {space_size}, diopi_dtype_uint8);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    impl::camb::KernelNms(k_dim,
                          k_type,
                          queue,
                          data_type_input,
                          boxes_t.data(),
                          scores.data(),
                          input_num_boxes,
                          max_output_boxes,
                          iou_threshold,
                          offset,
                          workspace.data(),
                          output_size.data(),
                          output.data());

    int bytes = sizeof(int) * output_size.numel();
    std::unique_ptr<char> output_size_cpu(new char[bytes]);
    cnrtMemcpyAsync(output_size_cpu.get(), output_size.data(), bytes, impl::camb::getStream(ctx), cnrtMemcpyDevToHost);
    impl::camb::syncStreamInCtx(ctx);
    int output_num = reinterpret_cast<int *>(output_size_cpu.get())[0];

    auto temp_out = impl::camb::requiresTensor(ctx, {output_num}, output.dtype());
    DIOPI_CALL(diopiSlice(ctx, diopiTensorHandle_t(temp_out), diopiTensorHandle_t(output), 0, 0, output_num, 1));

    *out_ = diopiTensorHandle_t(temp_out);
    return diopiSuccess;
}
