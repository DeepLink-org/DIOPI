cmake_minimum_required(VERSION 3.4)
project(ascend_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()


if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ascend-toolkit exists:" ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ASCEND_DIR:" ${ASCEND_DIR})
  include_directories(${ASCEND_DIR}/ascend-toolkit/latest/include/)
  link_directories(${ASCEND_DIR}/ascend-toolkit/latest/lib64)
else()
    message(FATAL_ERROR "No ascend-toolkit found.")
endif()

execute_process(
    COMMAND sh -c "cd ${CMAKE_CURRENT_SOURCE_DIR}/; git clone  https://gitee.com/ascend/op-plugin/ op-plugin || cd op-plugin && git checkout 7329adc23e88c713fa9a8fb3fe866794512d0b89 "
    OUTPUT_STRIP_TRAILING_WHITESPACE)

#file(GLOB_RECURSE OP_PLUGIN_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} op-plugin/op_plugin/ops/base_ops/aclops/*.cpp)
set(OP_PLUGIN_SRC op-plugin/op_plugin/ops/base_ops/aclops/AbsKernelNpu.cpp)
include_directories(op-plugin torch_npu ${CMAKE_CURRENT_SOURCE_DIR})
set(IMPL_SRC )

list(APPEND IMPL_SRC ${OP_PLUGIN_SRC})

if(NOT DEFINED PYTORCH_DIR)
  execute_process(
    COMMAND sh -c "dirname $(python -c 'import torch;print(torch.__path__[0])')"
    OUTPUT_VARIABLE PYTORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
set(TORCH_LIBRARY_DIR "${PYTORCH_DIR}/torch/lib")
link_directories(${TORCH_LIBRARY_DIR})
list(APPEND TORCH_INCLUDE_DIRS ${PYTORCH_DIR}/torch/include/
     ${PYTORCH_DIR}/torch/include/torch/csrc/api/include/)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")


# adaptor
set(USE_ADAPTOR OFF)
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(USE_ADAPTOR ON)
endif()

if(USE_ADAPTOR)

    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=ascend
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND IMPL_SRC ${GEN_FILES} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp)
    add_definitions(-DTEST_USE_ADAPTOR)
endif()

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ascendcl acl_op_compiler)

if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    #add_subdirectory(test)
endif()
