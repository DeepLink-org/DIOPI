project(torch_impl)

# muxi torch config
add_compile_definitions(USE_MACA=1)
set(USE_MACA ON)

find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
    add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
    add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})
    add_definitions(-DTORCH_VERSION=${Torch_VERSION})
    message(STATUS "Found Torch Version: ${Torch_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(BASE_TORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../torch")

file(GLOB REAL_IMPL_SRC
  ${BASE_TORCH_DIR}/functions/error.cpp
  ${BASE_TORCH_DIR}/functions/functions.cpp 

  ${BASE_TORCH_DIR}/functions/functions_lightllm.cpp 
  ${BASE_TORCH_DIR}/functions/functions_mmcv.cpp
  ${BASE_TORCH_DIR}/helper.cpp
  ${BASE_TORCH_DIR}/functions/functions_mmcv/*.cu

  # need discuss
  # ${BASE_TORCH_DIR}/functions/functions_ext.cpp 
  # ${BASE_TORCH_DIR}/functions/functions_ext/*.cu 
)

# adaptor
set(USE_ADAPTOR ON)
if(NOT EXISTS "${BASE_TORCH_DIR}/convert_config.yaml")
    message(FATAL_ERROR "convert_config.yaml doesn't exist.")
endif()

if(USE_ADAPTOR)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${BASE_TORCH_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=torch
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND REAL_IMPL_SRC ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/composite_ops.cpp)
endif()

# to muxi pytorch 
cuda_add_library(${DEVICEIMPL} SHARED ${REAL_IMPL_SRC})
target_link_libraries(${DEVICEIMPL} ${TORCH_LIBRARIES})


if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()
