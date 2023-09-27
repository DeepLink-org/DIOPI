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


file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)
list(APPEND IMPL_SRC ascend_tensor.cpp)

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")

target_link_libraries(${DEVICEIMPL} ascendcl acl_op_compiler)
target_include_directories(${DEVICEIMPL} SYSTEM PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})

if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../adaptor/codegen")
    add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
        --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_GEN_PATH}/../csrc/
        --config_device=ascend)
    set(USE_ADAPTOR "true")
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
    add_definitions(-DTEST_USE_ADAPTOR)
else()
    set(USE_ADAPTOR "false")
endif()

if (TEST)
    add_subdirectory(test)
endif()
