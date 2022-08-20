cmake_minimum_required(VERSION 3.4)
project(diopi_conformance_test)

option(USE_CXX "Whether to use cxx impl, or use c impl" ON)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/diopirt/include")
include_directories(include)

SET(DIOPIRT diopirt)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if (USE_CXX)
    set(CXX_LITERT_SRC csrc/litert.cpp)
    message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})
    add_definitions(-std=c++11)
    add_library(${DIOPIRT} SHARED ${CXX_LITERT_SRC})
else()
    set(C_LITERT_SRC csrc/litert.c)
    message("C_LITERT_SRC:" ${C_LITERT_SRC})
    add_library(${DIOPIRT} SHARED ${C_LITERT_SRC})
    set_target_properties(${DIOPIRT} PROPERTIES LINKER_LANGUAGE C)
endif()
set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")

set(DEVICEIMPL device_impl)
add_subdirectory(impl/cuda)
