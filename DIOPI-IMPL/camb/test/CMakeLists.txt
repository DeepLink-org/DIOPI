set(DIOPIRT diopirt)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")

set(TEST_SRC
    litert.cpp
    conform_test.cpp)
message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

add_library(${DIOPIRT} SHARED ${TEST_SRC})
set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")
target_link_libraries(${DIOPIRT} cnnl cnmlrt)

# Since all the symbols are declared weak in diopi_impl,
# depending on the compiler and linker version,
# linking `diopi_impl.so` may fail.
# Here, we use no-as-needed to ensure that diopi_impl.so is linked.
target_link_libraries(${DIOPIRT} -Wl, --no-as-needed diopi_impl -Wl, --as-needed)
