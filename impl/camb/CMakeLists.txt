cmake_minimum_required(VERSION 3.14)
project(camb_impl)

option(TEST "whether to test by using conformance test" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Neuware 6 REQUIRED)
if (NEUWARE_FOUND)
    set(USE_CAMB ON)
    set(CMAKE_CNCC_FLAGS "${CMAKE_CNCC_FLAGS};-fPIC;--bang-device-only")
    message(STATUS "Cambricon Neuware Root: ${NEUWARE_ROOT_DIR}")
    message(STATUS "Cambricon CNRT Version: ${CNRT_VERSION}")
    message(STATUS "Cambricon CNNL Version: ${CNNL_VERSION}")
    message(STATUS "Cambricon CNCC Flags: ${CMAKE_CNCC_FLAGS}")
else()
    message(FATAL_ERROR "No Cambricon Neuware toolkit found.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
  )
find_package(BANG)
if(BANG_FOUND)
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS};-fPIC;-std=c++14;-pthread;--bang-mlu-arch=mtp_220;--bang-mlu-arch=mtp_270;--bang-mlu-arch=mtp_290;--bang-mlu-arch=mtp_372;--bang-mlu-arch=mtp_592;-O3;-Wall")
  message(STATUS "BANG_CNCC_FLAGS: ${BANG_CNCC_FLAGS}")
else()
  message(FATAL_ERROR "BANG cannot be found.")
endif()

file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)
file(GLOB_RECURSE IMPL_SRC_MMCV RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions_mmcv/*.cpp functions/*.mlu )
file(GLOB_RECURSE IMPL_SRC_MMCV_MLU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions_mmcv/*.mlu)
list(APPEND IMPL_SRC mlu_ops_helper.cpp cnnl_helper.cpp diopi_helper.cpp init.cpp)
list(APPEND IMPL_SRC ${IMPL_SRC_MMCV})
list(APPEND IMPL_SRC ${IMPL_SRC_MMCV_MLU})

# adaptor
set(USE_ADAPTOR OFF)
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(USE_ADAPTOR ON)
endif()

if(USE_ADAPTOR)
    # dependency
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=camb
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND IMPL_SRC ${GEN_FILES} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp)
endif()


BANG_ADD_LIBRARY(${DEVICEIMPL} SHARED ${IMPL_SRC})
# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)

set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} cndev cnrt cnnl cnmlrt mluops)
target_include_directories(${DEVICEIMPL} SYSTEM PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})
if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    add_subdirectory(test)
endif()
