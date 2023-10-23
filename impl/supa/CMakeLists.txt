cmake_minimum_required(VERSION 3.4)
project(supa_impl)

option(DEBUG "whether to use debug" OFF)

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

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
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=supa
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND IMPL_SRC ${GEN_FILES} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp)
endif()


if(DEFINED ENV{DIOPI_SUPA_DIR})
  set(DIOPI_SUPA_DIR $ENV{DIOPI_SUPA_DIR})
else()
  set(DIOPI_SUPA_DIR /usr/local/supa)
endif()

find_library(DIOPI_BR_LIB NAMES ${DEVICEIMPL} HINTS ${DIOPI_SUPA_DIR}/lib ${DIOPI_SUPA_DIR}/lib64)
message(STATUS "DIOPI-IMPL lib: ${DIOPI_BR_LIB}")

if(NOT DIOPI_BR_LIB)
    message(FATAL_ERROR "${DEVICEIMPL} library not found !")
endif()

# ln -s /*/libdiopi_impl.so impl/lib/libdiopi_impl.so
execute_process(COMMAND ln -s ${DIOPI_BR_LIB} ${LIBRARY_OUTPUT_PATH}/lib${DEVICEIMPL}.so)

if (TEST)
  add_subdirectory(test)
endif()
