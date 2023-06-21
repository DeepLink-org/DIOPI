set(DIOPIRT diopirt)

# set path of SUPA runtime.
if(NOT DEFINED ENV{SUPA_PATH})
  set(SUPA_PATH /usr/local/supa)
else()
  set(SUPA_PATH $ENV{SUPA_PATH})
endif()

find_path(SUPA_INCLUDE supa.h
  ${SUPA_PATH}/include
)

if(NOT SUPA_INCLUDE)
  message(FATAL_ERROR " Could not find SUPA header files. Make sure fullstack is installed.")
else()
  message(STATUS " SUPA include path: ${SUPA_INCLUDE}")
endif()

find_library(SUPA_RUNTIME_LIB supa-runtime ${SUPA_PATH}/lib ${SUPA_PATH}/lib64)
if(NOT SUPA_RUNTIME_LIB)
  message(FATAL_ERROR " Could not find SUPA runtime lib. Make sure fullstack is installed.")
else()
  message(STATUS " SUPA runtime lib: ${SUPA_RUNTIME_LIB}")
endif()


include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM ${SUPA_INCLUDE})

set(TEST_SRC
    litert.cpp
    conform_test.cpp)

add_library(${DIOPIRT} SHARED ${TEST_SRC})
target_link_libraries(${DIOPIRT} ${SUPA_RUNTIME_LIB})
