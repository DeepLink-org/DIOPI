cmake_minimum_required(VERSION 3.4)
project(diopi_conformance_test)

option(ATTR_WEAK "Whether to use weak symbol for diopi functions" ON)
option(DEBUG "whether to use debug" OFF)

include_directories(SYSTEM "proto/include")
include_directories(include)

SET(DIOPIRT diopirt)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(DST_SRC_DIR ${CMAKE_SOURCE_DIR}/dist/src)

if (ATTR_WEAK)
    add_definitions(-DDIOPI_ATTR_WEAK)
endif()

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

add_definitions(-std=c++14)

include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/dist/include)

set(CXX_LITERT_SRC
    csrc/litert.cpp
)
message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

list(APPEND IMPL_CUDA "CUDA" "cuda")
list(APPEND IMPL_TORCH "TORCH" "LIBTORCH" "ATEN" "PYTORCH" "torch" "libtorch" "aten" "pytorch" "PyTorch")
list(APPEND IMPL_CAMB_TORCH "CAMB_PYTORCH" "camb_pytorch")
list(APPEND IMPL_CAMB "CAMB" "camb")
list(APPEND IMPL_ASCEND "ASCEND" "ascend")

if (${IMPL_OPT} IN_LIST IMPL_CUDA)
    find_package(CUDA REQUIRED)
    set(CXX_LITERT_SRC
        ${CXX_LITERT_SRC}
        csrc/cuda_conform_test.cpp
    )
    cuda_add_library(${DIOPIRT} SHARED ${CXX_LITERT_SRC})
    set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")
elseif (${IMPL_OPT} IN_LIST IMPL_TORCH)
    find_package(CUDA REQUIRED)
    set(CXX_LITERT_SRC
        ${CXX_LITERT_SRC}
        csrc/torch_conform_test.cpp
    )
    cuda_add_library(${DIOPIRT} SHARED ${CXX_LITERT_SRC})
    set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")
elseif (${IMPL_OPT} IN_LIST IMPL_CAMB_TORCH)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    find_package(Neuware 4.6 REQUIRED)
    set(CXX_LITERT_SRC
        ${CXX_LITERT_SRC}
        csrc/camb_torch_conform_test.cpp
    )
    add_library(${DIOPIRT} SHARED ${CXX_LITERT_SRC})
    set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")
elseif (${IMPL_OPT} IN_LIST IMPL_CAMB)
    set(CXX_LITERT_SRC
        ${CXX_LITERT_SRC}
        csrc/camb_conform_test.cpp
    )
    add_library(${DIOPIRT} SHARED ${CXX_LITERT_SRC})
elseif (${IMPL_OPT} IN_LIST IMPL_ASCEND)
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
    set(CXX_LITERT_SRC
        ${CXX_LITERT_SRC}
        csrc/ascend_conform_test.cpp
    )
    add_library(${DIOPIRT} SHARED ${CXX_LITERT_SRC})
else()
    message(WARNING "No implementation module is compiled, cmake requires option -DIMPL_OPT=CUDA or TORCH")
endif()

