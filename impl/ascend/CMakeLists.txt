cmake_minimum_required(VERSION 3.4)
project(ascend_impl)

option(DEBUG "whether to use debug" OFF)

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

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


file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)
#list(APPEND IMPL_SRC acl_helper.cpp)

# adaptor
set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../adaptor/codegen")
set(ADAPTOR_CSRC_PATH "${ADAPTOR_GEN_PATH}/../csrc")

set(AUTO_CONVERT "false")
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(AUTO_CONVERT "true")
endif()
add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
    --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH}
    --config_device=ascend --auto_convert=${AUTO_CONVERT})

file(GLOB_RECURSE ADAPTOR_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../adaptor/csrc/*.cpp)
list(APPEND IMPL_SRC ${ADAPTOR_SRC})

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ascendcl acl_op_compiler)
add_dependencies(${DEVICEIMPL} adaptor_code_gen)

if (TEST)
    add_subdirectory(test)
endif()