cmake_minimum_required(VERSION 3.14)
project(torch_impl)

option(HIP "Whether to use HIP when available" OFF)

include(cmake/TorchBaseFunc.cmake)
DIOPIFindTorch()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__;-std=c++17")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (DYLOAD)
    # creat an empty file to pass wrap_func.cpp's existence check
    # one can change code_gen.py or wrap_func.cpp to recompile once wrap_func.cpp built
    # execute_process(COMMAND touch ${CMAKE_CURRENT_SOURCE_DIR}/wrap_func.cpp)
    # add_custom_target(code_gen COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/code_gen.py
    #                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set(IMPL_SRC wrap_func.cpp)
endif()

file(GLOB REAL_IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/functions_mmcv/*.cu functions/functions_ext/*.cu functions/*.cpp helper.cpp)

# adaptor
set(USE_ADAPTOR ON)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    message(FATAL_ERROR "convert_config.yaml doesn't exist.")
endif()

if(USE_ADAPTOR)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${DIOPI_IMPL_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=torch
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND REAL_IMPL_SRC ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
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
        target_link_libraries(${DEVICEIMPL} -ldl)
    endif()
    add_dependencies(${DEVICEIMPL} code_gen) # make sure to recompile once code_gen.py changed
    target_link_libraries(${REALIMPL} torch)
    add_subdirectory(functions/functions_ext/flash-attention)
    target_link_libraries(${REALIMPL} diopi_torch_ext_flash_attn)
    target_link_libraries(${REALIMPL}  Threads::Threads)
    if (TEST)
        add_dependencies(${REALIMPL} diopirt)
        target_link_libraries(${REALIMPL} diopirt)
    endif()
else ()
    if(USE_HIP)
        hip_add_library(${DEVICEIMPL} SHARED ${REAL_IMPL_SRC})
        target_link_libraries(${DEVICEIMPL} ${HIP_LIBRARIES} ${TORCH_LIBRARIES})
    else()
        cuda_add_library(${DEVICEIMPL} SHARED ${REAL_IMPL_SRC})
        # target_compile_definitions(${DEVICEIMPL} PRIVATE __CUDA_NO_HALF_OPERATORS__)
        target_link_libraries(${DEVICEIMPL} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
        add_subdirectory(functions/functions_ext/flash-attention)
        target_link_libraries(${DEVICEIMPL} diopi_torch_ext_flash_attn)
    endif()
endif()

if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    add_subdirectory(test)
endif()
