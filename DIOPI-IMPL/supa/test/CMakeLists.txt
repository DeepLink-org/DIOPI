set(DIOPIRT diopirt)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM ${DIOPI_SUPA_DIR}/include)
include_directories(SYSTEM ${SUPA_PATH}/include)

find_library(SUPA_RUNTIME_LIB supa-runtime ${SUPA_PATH}/lib ${SUPA_PATH}/lib64)
message("SUPA runtime lib: ${SUPA_RUNTIME_LIB}")

set(TEST_SRC
    litert.cpp
    conform_test.cpp)

add_library(${DIOPIRT} SHARED ${TEST_SRC})
target_link_libraries(${DIOPIRT} ${SUPA_RUNTIME_LIB})
