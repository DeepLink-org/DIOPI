cmake_minimum_required(VERSION 3.4)
project(torch_cpp_impl)

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/lib/libtorch_1.12.1_cu113)
message(${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
if (Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14 -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS ${CMAKE_CXX_FLAGS})
else()
    message(FATAL_ERROR "no torch lib found")
endif()

set(IMPL_SRC
    conform_test.cpp
    functions.cpp
)

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ${DIOPIRT})
target_link_libraries(${DEVICEIMPL} ${CUDA_LIBRARIES})
target_link_libraries(${DEVICEIMPL} ${TORCH_LIBRARIES})
