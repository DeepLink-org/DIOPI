set(EXPORT_RT export_runtime)
set(DIOPI_FUNCTIONS export_functions)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 buildDIOPI_FUNCTIONS)

set(DIOPI_TEST_DIR "${CMAKE_SOURCE_DIR}/../diopi_test")

include_directories(SYSTEM "${DIOPI_TEST_DIR}/diopi_stub/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/diopi_stub/csrc")
set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/diopi_stub/codegen")

set(RUNTIME_SRC
    ${FUNCTION_SAVE_PATH}/litert.cpp
    conform_test.cpp
)
set(EXPORT_SRC
    ${FUNCTION_SAVE_PATH}/export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${EXPORT_RT} SHARED ${EXPORT_SRC})
add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(${EXPORT_RT} PRIVATE diopirt ${DEVICEIMPL})
target_link_libraries(diopirt cnrt cnmlrt cnnl)

file(GLOB TEST_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${DIOPI_TEST_DIR}/diopi_stub/codegen/*.py)
add_custom_target(test_gen_dependency DEPENDS ${TEST_TEMPLATE_CODE})

set(GEN_FILES ${FUNCTION_SAVE_PATH}/export_functions.cpp)
add_custom_target(test_code_gen ALL
    COMMAND python3 ${TEST_GEN_PATH}/gen.py --device=camb
    BYPRODUCTS ${GEN_FILES}
    DEPENDS test_gen_dependency)

set(FUNCTIONS_SRC ${GEN_FILES})

pybind11_add_module(${DIOPI_FUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPI_FUNCTIONS} PRIVATE diopirt -Wl,--no-as-needed ${DEVICEIMPL} -Wl,--as-needed)
add_dependencies(${DIOPI_FUNCTIONS} test_code_gen)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPI_FUNCTIONS}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${EXPORT_RT}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    DEPENDS ${DIOPI_FUNCTIONS} ${EXPORT_RT})
