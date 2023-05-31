cmake_minimum_required(VERSION 3.4)
project(supa_impl)

option(DEBUG "whether to use debug" OFF)

if (DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
else()
    SET(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED ENV{SUPA_DIOPI_DIR})
  set(SUPA_DIOPI_DIR $ENV{SUPA_DIOPI_DIR})
else()
  set(SUPA_DIOPI_DIR /usr/local/supa)
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
include_directories(${SUPA_DIOPI_DIR}/include)
link_directories(${SUPA_PATH}/lib ${SUPA_DIOPI_DIR}/lib)

# add biren so
target_link_libraries(${DEVICEIMPL} diopi_impl_br)
