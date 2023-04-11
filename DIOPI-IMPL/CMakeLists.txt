cmake_minimum_required(VERSION 3.4)
project(diopi_conformance_test)

option(ATTR_WEAK "Whether to use weak symbol for diopi functions" ON)
option(TEST "Whether to compile DIOPI with runtime" OFF)
option(DEBUG "whether to use debug" ON)
include_directories(SYSTEM "$ENV{DIOPI_PROTO_PATH}/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/proto/include")

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

if (ATTR_WEAK)
    add_definitions(-DDIOPI_ATTR_WEAK)
endif()

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_definitions(-DDIOPI_ATTR_WEAK)

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

set(DIOPI_PROTO_PATH ${CMAKE_SOURCE_DIR}/../DIOPI-PROTO)
set(DEVICEIMPL diopi_impl)

add_custom_command(OUTPUT copy_proto_cmd
    COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/proto ${DIOPI_PROTO_DIR}
    COMMAND ln -f ${DIOPI_PROTO_PATH}/include/ ${DIOPI_PROTO_DIR}/include/
)

list(APPEND IMPL_CUDA "CUDA" "cuda")
list(APPEND IMPL_TORCH "TORCH" "LIBTORCH" "ATEN" "PYTORCH" "torch" "libtorch" "aten" "pytorch" "PyTorch")
list(APPEND IMPL_CAMB_TORCH "CAMB_PYTORCH" "camb_pytorch")
list(APPEND IMPL_CAMB "CAMB" "camb")

if (${IMPL_OPT} IN_LIST IMPL_CUDA)
    add_subdirectory(cuda)
elseif (${IMPL_OPT} IN_LIST IMPL_TORCH)
    add_subdirectory(torch)
elseif (${IMPL_OPT} IN_LIST IMPL_CAMB_TORCH)
    add_subdirectory(camb_pytorch)
elseif (${IMPL_OPT} IN_LIST IMPL_CAMB)
    add_subdirectory(camb)
elseif (${IMPL_OPT} IN_LIST IMPL_ASCEND)
        add_subdirectory(ascend)
else()
    message(WARNING "No implementation module is compiled, cmake requires option -DIMPL_OPT=CUDA or TORCH")
endif()

