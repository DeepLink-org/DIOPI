cmake_minimum_required(VERSION 3.4)
project(cuda_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRING})
    message(STATUS "NVCC Path: ${CUDA_TOOLKIT_ROOT_DIR}/bin")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

    if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
        # cmake 3.3 and later version may auto-set nvcc host compiler with
        # -std=c++11 option, which will cause std redefine error
        if (${CMAKE_VERSION} VERSION_LESS "3.3")
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
        endif()

        if (${CMAKE_VERSION} VERSION_GREATER "3.3" AND ${CMAKE_VERSION} VERSION_LESS "4")
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
        endif()

        include(cmake/CUDAComputeArch.cmake)
        set(ARCH_HINTS)
        if (CUDA_ARCH_AUTO)
            LIST(APPEND ARCH_HINTS Auto)
        else()
            LIST(APPEND ARCH_HINTS Common)
        endif()
        cuda_select_nvcc_arch_flags(ARCH_FLAGS ${ARCH_HINTS})

        message(STATUS "Selected CUDA archs : ${ARCH_FLAGS}")
        string(REPLACE ";" " " ARCH_FLAG_STR "${ARCH_FLAGS}")
        string(REPLACE ";" " " CUDA_LIBRARIES_STR "${CUDA_LIBRARIES}")
        set(APPENDIX_VERSION_INFO ${APPENDIX_VERSION_INFO}
            "NVCC_PATH ${CUDA_TOOLKIT_ROOT_DIR}/bin"
            "CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}"
            "CUDA_LIBRARIES ${CUDA_LIBRARIES_STR}"
            "CUDA_ARCH_FLAGS ${ARCH_FLAG_STR}"
            "CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} --expt-extended-lambda)
    endif()
else()
    message(FATAL_ERROR "No suitable CUDA toolkit found.")
endif()

find_package(CUDNN REQUIRED)
if (CUDNN_FOUND)
    set(CUDNN_INCLUDE_PATH ${CUDNN_INCLUDE_DIR})
    set(CUDNN_LIBRARY_PATH ${CUDNN_LIBRARIES})
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Not found CUDNN.")
endif()

file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions_mmcv/*.cu)
list(APPEND IMPL_SRC functions.cpp)
list(APPEND IMPL_SRC functions.cu)
list(APPEND IMPL_SRC error.cpp)

cuda_add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")

target_link_libraries(${DEVICEIMPL} ${CUDA_LIBRARIES})
target_link_libraries(${DEVICEIMPL} ${CUDNN_LIBRARIES})

if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../DIOPI-ADAPTOR/codegen")
    add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
        --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${CMAKE_SOURCE_DIR}/../DIOPI-PROTO/include/diopi/
        --config_device=torch)
    set(USE_ADAPTOR "true")
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
    add_definitions(-DTEST_USE_ADAPTOR)
else()
    set(USE_ADAPTOR "false")
endif()

if (TEST)
    add_subdirectory(test)
endif()