project(muxi_impl)

# muxi torch config
add_compile_definitions(USE_MACA=1)
set(USE_MACA ON)

set(BASE_TORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../torch")
include(${BASE_TORCH_DIR}/cmake/TorchBaseFunc.cmake)
DIOPIFindTorch()

#TODO: Report bugs to muxi
# There are conflicts when muxi runtime used together with pip installed torch_cpu.
# so to use dipu with torch_cpu in muxi, maunual compile torch cpu with export BLAS=OpenBLAS.


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

file(GLOB REAL_IMPL_SRC
  ${BASE_TORCH_DIR}/functions/error.cpp
  ${BASE_TORCH_DIR}/functions/functions.cpp 

  ${BASE_TORCH_DIR}/functions/functions_lightllm.cpp 
  ${BASE_TORCH_DIR}/functions/functions_mmcv.cpp
  ${BASE_TORCH_DIR}/helper.cpp
  ${BASE_TORCH_DIR}/functions/functions_mmcv/*.cu

  ${BASE_TORCH_DIR}/functions/functions_ext.cpp 
  ${BASE_TORCH_DIR}/functions/functions_ext/*.cu 

  # mx cpp
  functions/functions.cpp 
)

# adaptor
set(USE_ADAPTOR ON)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/convert_config.yaml")
    message(FATAL_ERROR "convert_config.yaml doesn't exist.")
endif()

if(USE_ADAPTOR)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${DIOPI_IMPL_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH}
                        --config_device=muxi --base_device=torch
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND REAL_IMPL_SRC ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
endif()

cuda_add_library(${DEVICEIMPL} SHARED ${REAL_IMPL_SRC})
target_link_libraries(${DEVICEIMPL} ${TORCH_LIBRARIES})
add_subdirectory(functions/functions_ext/flash-attention)
target_link_libraries(${DEVICEIMPL} diopi_torch_ext_flash_attn)
target_include_directories(${DEVICEIMPL} PRIVATE ${BASE_TORCH_DIR})

if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    add_subdirectory(test)
endif()
