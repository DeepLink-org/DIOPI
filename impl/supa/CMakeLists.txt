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

find_library(DIOPI_BR_LIB NAMES ${DEVICEIMPL} HINTS ${DIOPI_SUPA_DIR}/lib)
message(STATUS "DIOPI-IMPL lib: ${DIOPI_BR_LIB}")
if(NOT DIOPI_BR_LIB)
    message(FATAL_ERROR "${DEVICEIMPL} library not found !")
endif()

file(COPY ${DIOPI_BR_LIB} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../lib/ FOLLOW_SYMLINK_CHAIN)
if (TEST)
  add_subdirectory(test)
endif()
