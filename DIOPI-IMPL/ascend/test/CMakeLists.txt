set(DIOPIRT diopirt)

set(TEST_SRC
    litert.cpp
    conform_test.cpp)
message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

add_library(${DIOPIRT} SHARED ${TEST_SRC})
set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")

