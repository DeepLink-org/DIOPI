cmake_minimum_required(VERSION 3.4)
project(camb_impl)

option(TEST "whether to test by using conformance test" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Neuware 4.6 REQUIRED)
if (NEUWARE_FOUND)
    set(USE_CAMB ON)
    set(CMAKE_CNCC_FLAGS "${CMAKE_CNCC_FLAGS};-fPIC;--bang-device-only")
    message(STATUS "Cambricon Neuware Root: ${NEUWARE_ROOT_DIR}")
    message(STATUS "Cambricon Neuware Home: ${NEUWARE_HOME}")
    message(STATUS "Cambricon Neuware Version: ${NEUWARE_VERSION}")
    message(STATUS "Cambricon CNCC Flags: ${CMAKE_CNCC_FLAGS}")
    message(STATUS "Cambricon CNCC Targets: ${CNCC_TARGETS}")
else()
    message(FATAL_ERROR "No Cambricon Neuware toolkit found.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
  )
find_package(BANG)
if(BANG_FOUND)
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS};-fPIC;-pthread;--bang-mlu-arch=mtp_220;--bang-mlu-arch=mtp_270;--bang-mlu-arch=mtp_290;--bang-mlu-arch=mtp_372;--bang-mlu-arch=mtp_592;-Wall")
  message(STATUS "BANG_CNCC_FLAGS: ${BANG_CNCC_FLAGS}")
else()
  message(FATAL_ERROR "BANG cannot be found.")
endif()

file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)
file(GLOB_RECURSE IMPL_SRC_MMCV RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions_mmcv/*.cpp)
file(GLOB_RECURSE IMPL_SRC_MMCV_MLU RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions_mmcv/*.mlu)
list(APPEND IMPL_SRC cnnl_helper.cpp)
list(APPEND IMPL_SRC ${IMPL_SRC_MMCV})
list(APPEND IMPL_SRC ${IMPL_SRC_MMCV_MLU})

if (TEST)
    add_subdirectory(test)
endif()

BANG_ADD_LIBRARY(${DEVICEIMPL} SHARED ${IMPL_SRC})

# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)

set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")

target_link_libraries(${DEVICEIMPL} cndev cnrt cnnl cnmlrt)
target_include_directories(${DEVICEIMPL} PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})
