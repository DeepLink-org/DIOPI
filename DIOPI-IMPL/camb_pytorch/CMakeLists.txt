cmake_minimum_required(VERSION 3.4)
project(torch_cpp_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Neuware 4.6 REQUIRED)

if (NEUWARE_VERSION VERSION_GREATER 6.0.0)
    set(TorchMLU_DIR /mnt/lustre/share/platform/env/miniconda3.6/envs/camb_pytorch1.6/lib/python3.6/site-packages/torch_mlu/share/cmake/TorchMLU/)
    set(Torch_DIR /mnt/lustre/share/platform/env/miniconda3.6/envs/camb_pytorch1.6/lib/python3.6/site-packages/torch/share/cmake/Torch/)
else ()
    set(TorchMLU_DIR /mnt/lustre/share/platform/env/miniconda3.6/envs/camb_pytorch1.6_2.4.1/lib/python3.6/site-packages/torch_mlu/share/cmake/TorchMLU/)
    set(Torch_DIR /mnt/lustre/share/platform/env/miniconda3.6/envs/camb_pytorch1.6_2.4.1/lib/python3.6/site-packages/torch/share/cmake/Torch/)
endif()

# set(Torch_DIR /mnt/lustre/huqingqing/cambricon_pytorch/pytorch/src/pytorch/torch/share/cmake/Torch/)

find_package(TorchMLU)
find_package(Torch)


if (NEUWARE_FOUND)
    set(USE_CAMB ON)
    set(CMAKE_CNCC_FLAGS "${CMAKE_CNCC_FLAGS};-fPIC;--bang-device-only")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    message(STATUS "Cambricon Neuware Root: ${NEUWARE_ROOT_DIR}")
    message(STATUS "Cambricon Neuware Version: ${NEUWARE_VERSION}")
    message(STATUS "Cambricon CNCC Flags: ${CMAKE_CNCC_FLAGS}")
    message(STATUS "Cambricon CNCC Targets: ${CNCC_TARGETS}")
else()
    message(FATAL_ERROR "No Cambricon Neuware toolkit found.")
endif()

if (TorchMLU_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14 -fPIC")
    message(STATUS "CXX compile options: ${CMAKE_CXX_FLAGS}")
else()
    message(FATAL_ERROR "Libtorch_mlu version is required, no suitable version was found")
endif()

set(IMPL_SRC
    conform_test.cpp
    error.cpp
    functions.cpp
)

link_directories(${NEUWARE_ROOT_DIR}/lib/llvm-mm/lib)
link_directories(${NEUWARE_ROOT_DIR}/lib64)
link_directories(${TORCH_INSTALL_PREFIX}/lib/)

include_directories(${TORCH_INCLUDE_DIRS})
add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
target_link_libraries(${DEVICEIMPL} ${TORCH_LIBRARIES} ${TORCH_MLU_LIBRARIES})


message(STATUS ".so used: ${TORCH_MLU_LIBRARIES}")
