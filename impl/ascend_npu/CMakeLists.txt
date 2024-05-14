cmake_minimum_required(VERSION 3.20)
project(ascend_impl)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
message(STATUS "downloading op-plugin")
FetchContent_Declare(op_plugin
  GIT_REPOSITORY https://gitee.com/ascend/op-plugin
  GIT_TAG 1efe08bb940ed85c28cc187f9c4ff7caafd6d5b1 # v5.0.1-pytorch2.1.0
)
FetchContent_MakeAvailable(op_plugin)
message(STATUS "op-plugin download done")

add_custom_target(op_plugin_gen
  COMMAND cd ${op_plugin_SOURCE_DIR} && bash ./gencode.sh 2.1 python
  BYPRODUCTS ${op_plugin_SOURCE_DIR}/op_plugin/OpInterface.h ${op_plugin_SOURCE_DIR}/op_plugin/OpInterface.cpp
             ${op_plugin_SOURCE_DIR}/op_plugin/AclOpsInterface.h  ${op_plugin_SOURCE_DIR}/op_plugin/OpApiInterface.h
)
add_subdirectory(third_party/acl)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acl/inc)

# find python3
find_package(Python COMPONENTS Interpreter Development)
MESSAGE(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
MESSAGE(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
include_directories(SYSTEM ${Python_INCLUDE_DIRS})
link_directories(${Python_LIBRARIES})

# find torch
if(NOT DEFINED PYTORCH_DIR)
  execute_process(
    COMMAND sh -c "python -c 'import torch;print(torch.__path__[0])'"
    OUTPUT_VARIABLE PYTORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(Torch REQUIRED
    PATHS  ${PYTORCH_DIR}
)

execute_process(
    COMMAND python -c "import torch;import re;version_tuple=re.search('[0-9\.]+', torch.__version__).group().split('.');factor=[10000,100, 1];DIOPI_TORCH_VERSION=sum([factor[i] * int(version_tuple[i]) for i in range(len(version_tuple))]);print(DIOPI_TORCH_VERSION)"
    OUTPUT_VARIABLE DIOPI_TORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(DIOPI_TORCH_VERSION=${DIOPI_TORCH_VERSION})
message(STATUS "TORCH_INSTALL_PREFIX:" ${TORCH_INSTALL_PREFIX})
message(STATUS "TORCH_CXX_FLAGS:" ${TORCH_CXX_FLAGS})   # include -D_GLIBCXX_USE_CXX11_ABI
message(STATUS "TORCH_INCLUDE_DIRS:" ${TORCH_INCLUDE_DIRS})
message(STATUS "DIOPI_TORCH_VERSION:" ${DIOPI_TORCH_VERSION})

include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib)

# Because many interfaces have not yet been implemented, -Wreturn-type=0 option is temporarily added.
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wno-return-type -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-variable "
)

set(OP_PLUGIN_BASE_OPS_DIR ${op_plugin_SOURCE_DIR}/op_plugin/ops/base_ops/)
set(OP_PLUGIN_UTILS_DIR ${op_plugin_SOURCE_DIR}/op_plugin/utils/)
set(OP_PLUGIN_SRC_V_DIR  ${op_plugin_SOURCE_DIR}/op_plugin/ops/v2r1/)

file(GLOB_RECURSE OP_PLUGIN_SRC
  ${OP_PLUGIN_BASE_OPS_DIR}/aclops/*.cpp
  ${OP_PLUGIN_BASE_OPS_DIR}/opapi/*.cpp
  ${OP_PLUGIN_UTILS_DIR}/*.cpp
  ${OP_PLUGIN_UTILS_DIR}/custom_functions/*.cpp
  ${OP_PLUGIN_SRC_V_DIR}/aclops/*.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/*.cpp
)

list(REMOVE_ITEM OP_PLUGIN_SRC
  ${OP_PLUGIN_BASE_OPS_DIR}/aclops/LstmCellKernelNpu.cpp
  ${OP_PLUGIN_BASE_OPS_DIR}/aclops/CtcLossKernelNpu.cpp
  ${OP_PLUGIN_BASE_OPS_DIR}/opapi/LstmCellKernelNpuApi.cpp
  ${OP_PLUGIN_BASE_OPS_DIR}/aclops/TriangularSolveKernelNpu.cpp
  ${OP_PLUGIN_SRC_V_DIR}/aclops/EmbeddingBagBackwardKernelNpu.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/StdKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/StdMeanKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/VarKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/TraceKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/MedianKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/EmbeddingBagBackwardKernelNpu.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/ForeachPowKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/ForeachPowScalarKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/RepeatInterLeaveKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/ZerosKernelNpuOpApi.cpp
  ${OP_PLUGIN_SRC_V_DIR}/opapi/ScatterUpdateKernelNpuOpApi.cpp
  ${OP_PLUGIN_BASE_OPS_DIR}/opapi/CtcLossKernelNpuOpApi.cpp
  ${OP_PLUGIN_UTILS_DIR}/AdvancedIndex.cpp
  ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/TriangularSolveKernelNpu.cpp
  ${OP_PLUGIN_UTILS_DIR}/custom_functions/opapi/TriangularSolveKernelNpuOpApi.cpp
)

set(ASCEND_IMPL_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_npu/csrc/impl/
)

set(DIOPI_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/diopi_impl/)
file(GLOB DIOPI_IMPL_SRC "${DIOPI_IMPL_DIR}/*.cpp" "${DIOPI_IMPL_DIR}/functions_ext/*.cpp")

add_definitions(-DBUILD_LIBTORCH)


include_directories(SYSTEM ${op_plugin_SOURCE_DIR} torch_npu ${CMAKE_CURRENT_SOURCE_DIR})

set(IMPL_SRC
  torch_npu/csrc/DIOPIAdapter.cpp
  torch_npu/csrc/AclOpCompileInterface.cpp
  torch_npu/csrc/NpuVariables.cpp
  torch_npu/csrc/AdvancedIndex.cpp
  torch_npu/csrc/NPUNativeFunctions.cpp
  torch_npu/csrc/CopyKernel.cpp
  torch_npu/csrc/CopyKernelOpApi.cpp
)

set(OLD_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ascend/)
set(OLD_IMPL_SRC
    ${OLD_IMPL_DIR}/ascend_tensor.cpp
    ${OLD_IMPL_DIR}/functions/abs.cpp
    ${OLD_IMPL_DIR}/functions/ceil.cpp
    ${OLD_IMPL_DIR}/functions/bmm.cpp
    ${OLD_IMPL_DIR}/functions/col2im.cpp
    ${OLD_IMPL_DIR}/functions/embedding.cpp
    ${OLD_IMPL_DIR}/functions/cumsum.cpp
    ${OLD_IMPL_DIR}/functions/fill.cpp
    ${OLD_IMPL_DIR}/functions/cat.cpp
    ${OLD_IMPL_DIR}/functions/pow.cpp
    ${OLD_IMPL_DIR}/functions/masked_fill.cpp
    ${OLD_IMPL_DIR}/functions/topk.cpp
    ${OLD_IMPL_DIR}/functions/baddbmm.cpp
    ${OLD_IMPL_DIR}/functions/conv2d.cpp
    ${OLD_IMPL_DIR}/functions/transpose.cpp
    ${OLD_IMPL_DIR}/functions/copy.cpp
    ${OLD_IMPL_DIR}/functions/normal.cpp
    ${OLD_IMPL_DIR}/functions/sort.cpp
    ${OLD_IMPL_DIR}/functions/isnan.cpp
    ${OLD_IMPL_DIR}/functions/activation.cpp
    ${OLD_IMPL_DIR}/functions/loss.cpp
    ${OLD_IMPL_DIR}/functions/contiguous.cpp
    ${OLD_IMPL_DIR}/functions/bitwise.cpp
    ${OLD_IMPL_DIR}/functions/upsample.cpp
    ${OLD_IMPL_DIR}/functions/tril.cpp
    ${OLD_IMPL_DIR}/functions/floor.cpp
    ${OLD_IMPL_DIR}/functions/mm.cpp
    ${OLD_IMPL_DIR}/functions/addcdiv.cpp
    ${OLD_IMPL_DIR}/functions/logic.cpp
    ${OLD_IMPL_DIR}/functions/linear.cpp
    ${OLD_IMPL_DIR}/functions/error.cpp
    ${OLD_IMPL_DIR}/functions/scatter.cpp
    ${OLD_IMPL_DIR}/functions/sgn.cpp
    ${OLD_IMPL_DIR}/functions/cast.cpp
    ${OLD_IMPL_DIR}/functions/sin.cpp
    ${OLD_IMPL_DIR}/functions/one_hot.cpp
    ${OLD_IMPL_DIR}/functions/triu.cpp
    ${OLD_IMPL_DIR}/functions/binary.cpp
    ${OLD_IMPL_DIR}/functions/cos.cpp
    ${OLD_IMPL_DIR}/functions/addcmul.cpp
    ${OLD_IMPL_DIR}/functions/interpolate.cpp
    ${OLD_IMPL_DIR}/functions/threshold.cpp
    ${OLD_IMPL_DIR}/functions/nonzero.cpp
    ${OLD_IMPL_DIR}/functions/stack.cpp
    ${OLD_IMPL_DIR}/functions/slice.cpp
    ${OLD_IMPL_DIR}/functions/unary.cpp
    ${OLD_IMPL_DIR}/functions/argmax.cpp
    ${OLD_IMPL_DIR}/functions/clamp.cpp
    ${OLD_IMPL_DIR}/functions/dropout.cpp
    ${OLD_IMPL_DIR}/functions/arange.cpp
    ${OLD_IMPL_DIR}/functions/gather.cpp
    ${OLD_IMPL_DIR}/functions/layer_norm.cpp
    ${OLD_IMPL_DIR}/functions/index_select.cpp
    ${OLD_IMPL_DIR}/functions/repeat.cpp
    ${OLD_IMPL_DIR}/functions/group_norm.cpp
    ${OLD_IMPL_DIR}/functions/flip.cpp
    ${OLD_IMPL_DIR}/functions/pool.cpp
    ${OLD_IMPL_DIR}/functions/uniform.cpp
    ${OLD_IMPL_DIR}/functions/batch_norm.cpp
    ${OLD_IMPL_DIR}/functions/minmax.cpp
    ${OLD_IMPL_DIR}/functions/norm.cpp
    ${OLD_IMPL_DIR}/functions/reduce.cpp
    ${OLD_IMPL_DIR}/functions/where.cpp
    ${OLD_IMPL_DIR}/functions/multinomial.cpp
    ${OLD_IMPL_DIR}/functions/mul.cpp
    ${OLD_IMPL_DIR}/functions/addmm.cpp
    ${OLD_IMPL_DIR}/functions/expand.cpp
    ${OLD_IMPL_DIR}/functions/linspace.cpp
    ${OLD_IMPL_DIR}/functions/ones.cpp
    ${OLD_IMPL_DIR}/functions/split.cpp
    ${OLD_IMPL_DIR}/functions/zeros.cpp
    ${OLD_IMPL_DIR}/functions_mmcv/roi_align_npu.cpp
    ${OLD_IMPL_DIR}/functions_ext/rms_norm.cpp
    #${OLD_IMPL_DIR}/test/export_functions.cpp
    #${OLD_IMPL_DIR}/test/conform_test.cpp
    ${OLD_IMPL_DIR}/common/utils.cpp
    ${OLD_IMPL_DIR}/common/debug.cpp
    ${OLD_IMPL_DIR}/common/generator_helper.cpp
    ${OLD_IMPL_DIR}/common/stream_lock.cpp
    ${OLD_IMPL_DIR}/env_vars.cpp
    ${OLD_IMPL_DIR}/init.cpp
)

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()

if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ascend-toolkit exists:" ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ASCEND_DIR:" ${ASCEND_DIR})
  include_directories(SYSTEM ${ASCEND_DIR}/ascend-toolkit/latest/include)
  include_directories(SYSTEM ${ASCEND_DIR}/ascend-toolkit/latest/include/aclnn)
  link_directories(${ASCEND_DIR}/ascend-toolkit/latest/lib64)
else()
    message(FATAL_ERROR "No ascend-toolkit found.")
endif()

list(APPEND IMPL_SRC ${OP_PLUGIN_SRC} ${DIOPI_IMPL_SRC} ${OLD_IMPL_SRC})

# adaptor
set(USE_ADAPTOR ON)
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(USE_ADAPTOR ON)
endif()

if(USE_ADAPTOR)
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../ascend/ ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=ascend --impl_plugin=True
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND IMPL_SRC ${GEN_FILES} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp)
    add_definitions(-DTEST_USE_ADAPTOR)
endif()

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()

if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ascend-toolkit exists:" ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ASCEND_DIR:" ${ASCEND_DIR})
  #include_directories(${ASCEND_DIR}/ascend-toolkit/latest/include/)
  link_directories(${ASCEND_DIR}/ascend-toolkit/latest/lib64)
else()
    message(FATAL_ERROR "No ascend-toolkit found.")
endif()


# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
add_dependencies(${DEVICEIMPL} op_plugin_gen)
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_include_directories(${DEVICEIMPL} PRIVATE ${ASCEND_DIR}/ascend-toolkit/latest/include/aclnn)
target_include_directories(${DEVICEIMPL} SYSTEM PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})
target_link_libraries(${DEVICEIMPL} ascendcl acl_op_compiler graph c10 torch_cpu nnopbase opapi ${Python_LIBRARIES})

if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    add_subdirectory(test)
endif()
