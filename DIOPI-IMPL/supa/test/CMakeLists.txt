set(DIOPIRT diopirt)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM ${DIOPI_SUPA_DIR}/include)
link_directories(${SUPA_PATH}/lib ${DIOPI_SUPA_DIR}/lib)

set(TEST_SRC
    litert.cpp
    conform_test.cpp)
message("TEST_SRC:" ${TEST_SRC})

add_library(${DIOPIRT} SHARED ${TEST_SRC})
target_link_libraries(${DIOPIRT} supa-runtime)
