set(DIOPIRT export_runtime)
set(DIOPI_FUNCTIONS export_functions)

add_compile_options(-fno-elide-constructors -fpermissive)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

set(DIOPI_TEST_DIR "${CMAKE_SOURCE_DIR}/../diopi_test")

include_directories(SYSTEM "${DIOPI_TEST_DIR}/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")
include_directories(SYSTEM "~/miniconda3/envs/dipu/include/python3.8")

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/csrc")
set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/codegen")

set(RUNTIME_SRC
    ${FUNCTION_SAVE_PATH}/litert.cpp
    conform_test.cpp)

set(EXPORT_SRC
    ${FUNCTION_SAVE_PATH}/export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})


pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(${DIOPIRT} PRIVATE diopirt)
#target_link_libraries(diopirt ${DEVICEIMPL})
# Since all the symbols are declared weak in diopi_impl,
# depending on the compiler and linker version,
# linking `diopi_impl.so` may fail.
# Here, we use no-as-needed to ensure that diopi_impl.so is linked.
#target_link_libraries(${DIOPIRT} -Wl,--no-as-needed ${DEVICEIMPL} -Wl,--as-needed)
target_link_libraries(diopirt -Wl,--no-as-needed ${DEVICEIMPL} -Wl,--as-needed)

add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py --device=topsrider)

# file(MAKE_DIRECTORY ${FUNCTION_SAVE_PATH})
set(EXPORT_FUNCTIONS "${FUNCTION_SAVE_PATH}/export_functions.cpp")
file(TOUCH ${EXPORT_FUNCTIONS})
set(FUNCTIONS_SRC
    ${EXPORT_FUNCTIONS}
)

pybind11_add_module(${DIOPI_FUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPI_FUNCTIONS} PRIVATE -Wl,--no-as-needed diopirt -Wl,--as-needed)
add_dependencies(${DIOPI_FUNCTIONS} test_code_gen)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPI_FUNCTIONS}> ${CMAKE_SOURCE_DIR}/../diopi_test/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${CMAKE_SOURCE_DIR}/../diopi_test/python
    DEPENDS ${DIOPI_FUNCTIONS} ${DIOPIRT})

