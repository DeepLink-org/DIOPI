cmake_minimum_required(VERSION 3.4)
project(kunlunxin_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(XdnnTorch REQUIRED)
if (XDNNTORCH_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14 -fPIC")
    message(STATUS "CXX compile options: ${CMAKE_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No KLX XdnnTorch toolkit found.")
endif()

include_directories(${XDNNTORCH_INCLUDE_DIR})
include_directories(${XDNN_INCLUDE_DIR})
include_directories(${XPURT_INCLUDE_DIR})

file(GLOB_RECURSE IMPL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} functions/*.cpp common/*.cpp)
add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${XDNNTORCH_LIBRARIES} ${XDNN_LIBRARIES} ${XPURT_LIBRARIES})

if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    message(STATUS "using convert config: ${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../adaptor/codegen")
    add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
        --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${CMAKE_SOURCE_DIR}/../proto/include/diopi/
        --config_device=kunlunxin)
    set(USE_ADAPTOR "true")
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
    add_definitions(-DTEST_USE_ADAPTOR)
else()
    message(STATUS "not using adaptor")
    set(USE_ADAPTOR "false")
endif()

if (TEST)
    add_subdirectory(test)
endif()