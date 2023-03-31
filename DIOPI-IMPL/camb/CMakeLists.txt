cmake_minimum_required(VERSION 3.4)
project(camb_impl)

option(DEBUG "whether to use debug" OFF)

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Neuware 4.6 REQUIRED)
if (NEUWARE_FOUND)
    set(USE_CAMB ON)
    set(CMAKE_CNCC_FLAGS "${CMAKE_CNCC_FLAGS};-fPIC;--bang-device-only")
    message(STATUS "Cambricon Neuware Root: ${NEUWARE_ROOT_DIR}")
    message(STATUS "Cambricon Neuware Version: ${NEUWARE_VERSION}")
    message(STATUS "Cambricon CNCC Flags: ${CMAKE_CNCC_FLAGS}")
    message(STATUS "Cambricon CNCC Targets: ${CNCC_TARGETS}")
else()
    message(FATAL_ERROR "No Cambricon Neuware toolkit found.")
endif()

file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)
list(APPEND IMPL_SRC cnnl_helper.cpp)

if (RUNTIME)
    list(APPEND IMPL_SRC conform_test.cpp)
endif()

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})

# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)

set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
if (RUNTIME)
    target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
endif()
target_link_libraries(${DEVICEIMPL} cndev cnrt cnnl cnmlrt)
target_include_directories(${DEVICEIMPL} PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})
