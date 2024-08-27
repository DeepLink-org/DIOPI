project(muxi_impl)

# muxi torch config
add_compile_definitions(USE_MACA=1)
set(USE_MACA ON)

include(../cmake/ImplHelper.cmake)
include(../cmake/TorchBaseFunc.cmake)
diopi_find_torch()

# TODO: Report bugs to muxi
# There has conflict when muxi runtime used together with pip installed torch_cpu.
# so to use dipu with torch_cpu in muxi, maunual compile torch cpu with export BLAS=OpenBLAS.

set(BASE_TORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../torch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

file(GLOB REAL_IMPL_SRC
    ${BASE_TORCH_DIR}/functions/error.cpp
    ${BASE_TORCH_DIR}/functions/functions.cpp
    ${BASE_TORCH_DIR}/functions/functions_sparse.cpp
    ${BASE_TORCH_DIR}/functions/functions_lightllm.cpp
    ${BASE_TORCH_DIR}/functions/functions_mmcv.cpp
    ${BASE_TORCH_DIR}/helper.cpp
    ${BASE_TORCH_DIR}/functions/functions_mmcv/*.cu
    ${BASE_TORCH_DIR}/functions/functions_ext.cpp
    ${BASE_TORCH_DIR}/functions/functions_ext/*.cu
    ${BASE_TORCH_DIR}/functions/functions_sparse/*.cu
    ${BASE_TORCH_DIR}/build_aten.cpp
    # mx cpp
    functions/functions.cpp
)

# adaptor
set(USE_ADAPTOR ON)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/convert_config.yaml")
    message(FATAL_ERROR "convert_config.yaml doesn't exist.")
endif()

if(USE_ADAPTOR)
    diopi_use_adapter(${ADAPTOR_DIR} ${DIOPI_IMPL_DIR} "torch" "" REAL_IMPL_SRC)
endif()

if(${DYLOAD})
    prep_dyn_load(${DIOPI_IMPL_DIR} ${DEVICEIMPL})
    set(REAL_IMPL diopi_real_impl)
else()
    set(REAL_IMPL ${DEVICEIMPL})
endif()

cuda_add_library(${REAL_IMPL} SHARED ${REAL_IMPL_SRC})
target_link_libraries(${REAL_IMPL} ${TORCH_LIBRARIES})
add_subdirectory(functions/functions_ext/flash-attention)
target_link_libraries(${REAL_IMPL} diopi_torch_ext_flash_attn)
target_include_directories(${REAL_IMPL} PRIVATE ${BASE_TORCH_DIR})

if(USE_ADAPTOR)
    add_dependencies(${REAL_IMPL} adaptor_code_gen)
endif()
if(${DYLOAD})
    handle_dyn_torch(${DIOPI_IMPL_DIR} ${REAL_IMPL} ${TORCH_INSTALL_PREFIX} ${DEVICEIMPL})
endif()

if (TEST)
    add_subdirectory(test)
endif()
