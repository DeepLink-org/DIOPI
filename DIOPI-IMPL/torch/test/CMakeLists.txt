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
# Since all the symbols are declared weak in diopi_impl,
# depending on the compiler and linker version,
# linking `diopi_impl.so` may fail.
# Here, we use no-as-needed to ensure that diopi_impl.so is linked.
target_link_libraries(${DIOPIRT} -Wl,--no-as-needed ${DEVICEIMPL} -Wl,--as-needed)
