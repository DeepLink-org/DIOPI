cmake_minimum_required(VERSION 3.14)
project(torch_impl)

option(HIP "Whether to use HIP when available" OFF)
option(DIOPI_TORCH_UNSAFE_BUILDATEN "Whether to use fast but unsafe buildATen (caution: only use this with DIPU)" OFF)

include(../cmake/ImplHelper.cmake)
include(../cmake/TorchBaseFunc.cmake)
diopi_find_torch()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__;-std=c++17")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

file(GLOB REAL_IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/functions_mmcv/*.cu functions/functions_ext/*.cu functions/*.cpp helper.cpp build_aten.cpp)

# adaptor
set(USE_ADAPTOR ON)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    message(FATAL_ERROR "convert_config.yaml doesn't exist.")
endif()

if(USE_ADAPTOR)
  diopi_use_adapter("--config_device=torch")
endif()

prep_dyn_load(DYLOAD)

if(HIP)
    include(../cmake/LoadHIP.cmake)
    add_definitions(-DUSE_HIP)
    # hipify-perl
    execute_process(
        COMMAND bash ${HIP_PATH}/bin/hipconvertinplace-perl.sh ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # hipify-amend
    execute_process(
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/hipify-amend.sh
    )
    hip_add_library(${REAL_IMPL} SHARED ${REAL_IMPL_SRC})
    target_link_libraries(${REAL_IMPL} ${HIP_LIBRARIES} ${TORCH_LIBRARIES})

else()
    cuda_add_library(${REAL_IMPL} SHARED ${REAL_IMPL_SRC})
    # target_compile_definitions(${REAL_IMPL} PRIVATE __CUDA_NO_HALF_OPERATORS__)
    target_link_libraries(${REAL_IMPL} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
    add_subdirectory(functions/functions_ext/flash-attention)
    target_link_libraries(${REAL_IMPL} diopi_torch_ext_flash_attn)
endif()

if(USE_ADAPTOR)
    add_dependencies(${REAL_IMPL} adaptor_code_gen)
endif()

if(DIOPI_TORCH_UNSAFE_BUILDATEN)
    target_compile_definitions(${DEVICEIMPL} PRIVATE DIOPI_TORCH_UNSAFE_BUILDATEN)
endif()

if (TEST)
    add_subdirectory(test)
endif()
