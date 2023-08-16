cmake_minimum_required(VERSION 3.4)
project(tops_impl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if("${TOPS_HOME}" STREQUAL "")
    if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
        set(TOPS_HOME /root/diopi/tops/build/install/usr/)
    else()
        set(TOPS_HOME /root/diopi/tops/build/install/usr/)
    endif()
endif()

if("${DEVICEIMPL}" STREQUAL "")
    set(DEVICEIMPL libdiopi_impl)
endif()

if("${TOPS_HOME}" STREQUAL "")
    set(TOPS_HOME /usr/)
endif()

include_directories(
    ${TOPS_HOME}/include/
)

#add_link_options (-lz3)#

find_package(OPS QUIET)

if(OPS_FOUND)
    set(OPS_INCLUDE_PATH ${OPS_INCLUDE_DIR})
    set(OPS_LIBRARY_PATH ${OPS_LIBRARIES})
    include_directories(SYSTEM ${OPS_INCLUDE_DIR})
else()
    #add_subdirectory(ops ${CMAKE_CURRENT_SOURCE_DIR}/ops/build)
    add_subdirectory(ops)

    set(OPS_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ops/)
    set(OPS_LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ops/build/libops.so)
    include_directories(SYSTEM ${OPS_INCLUDE_DIR})
endif()

# file(GLOB OPS_SRCS  "ops/*.cpp")

set(IMPL_SRC test/conform_test.cpp functions.cpp ${OPS_SRCS})

# adaptor
set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../adaptor/codegen")
set(ADAPTOR_CSRC_PATH "${ADAPTOR_GEN_PATH}/../csrc")

set(AUTO_CONVERT "false")
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(AUTO_CONVERT "true")
endif()
add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
    --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH}
    --config_device=topsrider --auto_convert=${AUTO_CONVERT})

file(GLOB_RECURSE ADAPTOR_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../../adaptor/csrc/*.cpp)
list(APPEND IMPL_SRC ${ADAPTOR_SRC})

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})

add_dependencies(${DEVICEIMPL} adaptor_code_gen)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    target_compile_options(${DEVICEIMPL} PUBLIC -std=c++17 -O0 -ggdb3)
else()
    target_compile_options(${DEVICEIMPL} PUBLIC -std=c++17 -O3)
endif()

target_link_directories(${DEVICEIMPL} PUBLIC ${TOPS_HOME}/lib/ ${CMAKE_CURRENT_LIST_DIR}/prebuilt)

# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
target_include_directories(${DEVICEIMPL} SYSTEM PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})
#target_link_libraries(${DEVICEIMPL} efrt)
#target_link_libraries(${DEVICEIMPL} dtu_sdk)
target_link_libraries(${DEVICEIMPL} ops)

if (TEST)
    add_subdirectory(test)
endif()
