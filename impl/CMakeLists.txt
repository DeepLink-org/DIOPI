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

set(DIOPI_IMPL_DIR "${PROJECT_SOURCE_DIR}")
set(ADAPTOR_DIR "${DIOPI_IMPL_DIR}/../adaptor")
set(DIOPI_TEST_DIR "${DIOPI_IMPL_DIR}/../diopi_test")

if (TEST)
    # find the file Python.h and add it in the include path.
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "PYTHON_INCLUDE_DIRS: " ${PYTHON_INCLUDE_DIRS})
    include_directories(SYSTEM ${Python_INCLUDE_DIRS})

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

set(COMMON_FLAG "-Wall -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAG}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAG}")

set(DEVICEIMPL diopi_impl)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

list(APPEND IMPL_CUDA "CUDA" "cuda")
list(APPEND IMPL_TORCH "TORCH" "LIBTORCH" "ATEN" "PYTORCH" "torch" "libtorch" "aten" "pytorch" "PyTorch")
list(APPEND IMPL_MUXI "MUXI" "muxi")
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
elseif (${IMPL_OPT} IN_LIST IMPL_MUXI)
    add_subdirectory(muxi)
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

# 1.the lib ${DEVICEIMPL} in which all exported symbols are 'weak' can be considered as 'no-needed' lib. 
# some compilers force link such libs by default, but others having 'as-needed' default link-config may 
# throw away these libs. so we manually add '-no-as-needed' here to guarantee linking ${DEVICEIMPL}.
# eg: if you compiler don't link 'no-needed' libs by default, please use 'g++ -dumpspecs' and see '*link:' section
# to check if it contains policy like '%{!fsanitize=*:--as-needed}' or other policy having '--as-needed' set.
# you can change compiler's default spec by typing 'gcc -specs=./new.specs' but it's hard to use.
# Supplementary: https://gcc.gnu.org/onlinedocs/gcc/Spec-Files.html

# 2. when the code below adding "-no-as-needed" opt to link.txt, the opt isn't be added exactly before the place
# ${DEVICEIMPL} is linked but as a link option before any link-items. if another link-item change linking-policy
# as "-Wl,--no-as-needed,\"\$<TARGET_FILE:other_lib>\" -Wl,--as-needed" and ${DEVICEIMPL} is linked just after 
# this item; it will still be linked by -as-needed and finally be throw away by ld and cause error !!
# so if this error happens, please add link-option "-Wl,-no-as-needed" to the lib which link ${DEVICEIMPL}.

target_link_options(${DEVICEIMPL} INTERFACE "LINKER:-no-as-needed")

# install
install(DIRECTORY ${DIOPI_IMPL_DIR}/../proto/include/ TYPE INCLUDE)
install(FILES lib/lib${DEVICEIMPL}.so TYPE LIB)
