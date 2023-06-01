cmake_minimum_required(VERSION 3.4)
project(supa_impl)

option(DEBUG "whether to use debug" OFF)

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED ENV{DIOPI_SUPA_DIR})
  set(DIOPI_SUPA_DIR $ENV{DIOPI_SUPA_DIR})
else()
  set(DIOPI_SUPA_DIR /usr/local/supa)
endif()

if(NOT DEFINED ENV{SUPA_PATH})
    set(SUPA_PATH /usr/local/supa)
else()
    set(SUPA_PATH $ENV{SUPA_PATH})
endif()


file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)

if (TEST)
  add_subdirectory(test)
endif()

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
include_directories(${DIOPI_SUPA_DIR}/include)

find_library(DIOPI_BR_LIB NAMES diopi_impl_br HINTS ${DIOPI_SUPA_DIR}/lib)
message("BR-DIOPI LIB:${DIOPI_BR_LIB}")

# add biren so
target_link_libraries(${DEVICEIMPL} ${DIOPI_BR_LIB})
