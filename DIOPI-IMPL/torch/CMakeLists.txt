cmake_minimum_required(VERSION 3.4)
project(torch_cpp_impl)

find_package(Torch REQUIRED)
if (Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14 -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    list(APPEND CUDA_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__")
    message(STATUS "CXX compile options: ${CMAKE_CXX_FLAGS}")
    add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
    add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
    add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})
    add_definitions(-DTORCH_VERSION=${Torch_VERSION})
    message(STATUS "Found Torch Version: ${Torch_VERSION}")
else()
    message(FATAL_ERROR "No libtorch found")
endif()

set(IMPL_SRC
    conform_test.cpp
    functions.cpp
    nms_kernel.cu
    roi_align_kernel.cu
)

cuda_add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
target_link_libraries(${DEVICEIMPL} ${CUDA_LIBRARIES})
target_link_libraries(${DEVICEIMPL} ${TORCH_LIBRARIES})
# target_link_libraries(${DEVICEIMPL} ${CU_KERNEL_LIB})
