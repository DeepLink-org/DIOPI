cmake_minimum_required(VERSION 3.4)
project(torch_cpp_impl)

option(HIP "Whether to use HIP when available" OFF)

find_package(Torch 1.10 REQUIRED)
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
    message(FATAL_ERROR "Libtorch version 1.10 is required, no suitable version was found")
endif()

if (DYLOAD)
    # creat an empty file to pass wrap_func.cpp's existence check
    # one can change code_gen.py or wrap_func.cpp to recompile once wrap_func.cpp built
    execute_process(COMMAND touch ${CMAKE_CURRENT_SOURCE_DIR}/wrap_func.cpp)
    add_custom_target(code_gen COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/code_gen.py
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(IMPL_SRC error.cpp wrap_func.cpp)
endif()

set(REAL_IMPL_SRC
    error.cpp
    functions.cpp
    nms_kernel.cu
    roi_align_kernel.cu
)

if (TEST)
    add_subdirectory(test)
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-version-script=${CMAKE_CURRENT_SOURCE_DIR}/diopi.map")
endif()

if(HIP)
    include(../cmake/LoadHIP.cmake)
endif()

if(USE_HIP)
    add_definitions(-DUSE_HIP)
    # hipify-perl
    execute_process(
        COMMAND bash ${HIP_PATH}/bin/hipconvertinplace-perl.sh ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # hipify-amend
    execute_process(
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/hipify-amend.sh
    )
endif()

if (DYLOAD)
    set(REALIMPL diopi_real_impl)
    if(USE_HIP)
        hip_add_library(${REALIMPL} SHARED ${REAL_IMPL_SRC})
        add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
        target_link_libraries(${DEVICEIMPL} -ldl ${HIP_LIBRARIES})
    else()
        cuda_add_library(${REALIMPL} SHARED ${REAL_IMPL_SRC})
        add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
        target_link_libraries(${DEVICEIMPL} -ldl ${CUDA_LIBRARIES})
    endif()
    add_dependencies(${DEVICEIMPL} code_gen) # make sure to recompile once code_gen.py changed
    target_link_libraries(${REALIMPL} ${TORCH_LIBRARIES})
else ()
    if(USE_HIP)
        hip_add_library(${DEVICEIMPL} SHARED ${REAL_IMPL_SRC})
        target_link_libraries(${DEVICEIMPL} ${HIP_LIBRARIES} ${TORCH_LIBRARIES})
    else()
        cuda_add_library(${DEVICEIMPL} SHARED ${REAL_IMPL_SRC})
        target_link_libraries(${DEVICEIMPL} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
    endif()
endif()
