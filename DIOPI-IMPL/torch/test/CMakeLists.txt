set(DIOPIRT diopirt)
find_package(CUDA REQUIRED)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
message(${PROJECT_SOURCE_DIR})

set(TEST_SRC
    litert.cpp
    conform_test.cpp)
message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

cuda_add_library(${DIOPIRT} SHARED ${TEST_SRC})
set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")

