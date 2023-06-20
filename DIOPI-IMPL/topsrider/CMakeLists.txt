cmake_minimum_required(VERSION 3.4)
project(topsrider_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(TOPSDNN REQUIRED)
if(TOPSDNN_FOUND)
    set(TOPSDNN_INCLUDE_PATH ${TOPSDNN_INCLUDE_DIR})
    set(TOPSDNN_LIBRARY_PATH ${TOPSDNN_LIBRARIES})
    include_directories(SYSTEM ${TOPSDNN_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Not found TOPSDNN.")
endif()

message("TOPSDNN_INCLUDE_DIR:" ${TOPSDNN_INCLUDE_DIR}) 
find_package(TOPSRT REQUIRED)

if(TOPSRT_FOUND)
    set(TOPSRT_INCLUDE_PATH ${TOPSRT_INCLUDE_DIR})
    set(TOPSRT_LIBRARY_PATH ${TOPSRT_LIBRARIES})
    include_directories(SYSTEM ${TOPSRT_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Not found TOPSRT.")
endif()
message("TOPSRT_INCLUDE_DIR:" ${TOPSRT_INCLUDE_DIR})

set(IMPL_SRC conform_test.cpp functions.cpp error.cpp)

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
target_link_libraries(${DEVICEIMPL} ${TOPSDNN_LIBRARIES})
target_link_libraries(${DEVICEIMPL} ${TOPSRT_LIBRARIES})