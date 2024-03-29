cmake_minimum_required(VERSION 3.20)
project(diopi_impl)

option(TEST "Whether to compile DIOPI with runtime" OFF)
option(DYLOAD "Whether to load so in runtime by dlopen" OFF)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/proto/include")
include_directories("${PROJECT_SOURCE_DIR}/../adaptor/csrc")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "use cpp_17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add stdc++ but not gnuc++
set(CMAKE_CXX_EXTENSIONS OFF)

set (ADAPTOR_DIR "${CMAKE_SOURCE_DIR}/../adaptor")
set (DIOPI_TEST_DIR "${CMAKE_SOURCE_DIR}/../diopi_test")

if (TEST)
    # find the file Python.h and add it in the include path.
    execute_process(COMMAND  sh -c "dirname $(find  $(dirname $(which python))/../ -name Python.h)" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    if ("${PYTHON_INCLUDE_DIR}" STREQUAL "")
        message(FATAL_ERROR "no Python.h is found")
    endif()
    message(STATUS "PYTHON_INCLUDE_DIR: " ${PYTHON_INCLUDE_DIR})
    include_directories(${PYTHON_INCLUDE_DIR})
endif(TEST)

# use gcover
option(ENABLE_COVERAGE "Use gcov" OFF)
message(STATUS ENABLE_COVERAGE=${ENABLE_COVERAGE})

# the default is Release
if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(COMMON_FLAG "-Wall -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAG}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAG}")


set(DEVICEIMPL diopi_impl)

list(APPEND IMPL_CUDA "CUDA" "cuda")
list(APPEND IMPL_TORCH "TORCH" "LIBTORCH" "ATEN" "PYTORCH" "torch" "libtorch" "aten" "pytorch" "PyTorch")
list(APPEND IMPL_TOPS "TOPS" "tops" "TOPSRIDER" "topsrider")
list(APPEND IMPL_CAMB_TORCH "CAMB_PYTORCH" "camb_pytorch")
list(APPEND IMPL_CAMB "CAMB" "camb")
list(APPEND IMPL_ASCEND "ASCEND" "ascend")
list(APPEND IMPL_SUPA "SUPA" "supa")
list(APPEND IMPL_DROPLET "DROPLET" "droplet")
list(APPEND IMPL_KUNLUNXIN "KUNLUNXIN" "kunlunxin")

if (${IMPL_OPT} IN_LIST IMPL_CUDA)
    add_subdirectory(cuda)
elseif(${IMPL_OPT} IN_LIST IMPL_TOPS)
    add_subdirectory(topsrider)
elseif (${IMPL_OPT} IN_LIST IMPL_TORCH)
    add_subdirectory(torch)
elseif (${IMPL_OPT} IN_LIST IMPL_CAMB_TORCH)
    add_subdirectory(camb_pytorch)
elseif (${IMPL_OPT} IN_LIST IMPL_CAMB)
    add_subdirectory(camb)
elseif (${IMPL_OPT} IN_LIST IMPL_ASCEND)
    add_subdirectory(ascend_npu)
elseif (${IMPL_OPT} IN_LIST IMPL_SUPA)
    add_subdirectory(supa)
elseif (${IMPL_OPT} IN_LIST IMPL_DROPLET)
    add_subdirectory(droplet)
elseif (${IMPL_OPT} IN_LIST IMPL_KUNLUNXIN)
    add_subdirectory(kunlunxin)
else()
    message(WARNING "No implementation module is compiled, cmake requires option -DIMPL_OPT=CUDA or TORCH")
endif()

# install
install(DIRECTORY ${CMAKE_SOURCE_DIR}/../proto/include/ TYPE INCLUDE)
install(FILES lib/lib${DEVICEIMPL}.so TYPE LIB)
