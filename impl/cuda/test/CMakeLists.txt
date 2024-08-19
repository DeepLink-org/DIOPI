set(DIOPIRT diopirt)
find_package(CUDA REQUIRED)

set(DIOPIRT export_runtime)
set(DIOPIFUNCTIONS export_functions)
find_package(CUDA REQUIRED)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${DIOPI_IMPL_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/diopi_stub/csrc")
set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/diopi_stub/codegen")
set(RUNTIME_SRC
    litert.cpp
    conform_test.cpp
)

set(EXPORT_SRC
    export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
cuda_add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(${DIOPIRT} PRIVATE diopirt)
target_link_libraries(diopirt ${DEVICEIMPL})

file(GLOB TEST_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${DIOPI_TEST_DIR}/diopi_stub/codegen/*.py)
add_custom_target(test_gen_dependency DEPENDS ${TEST_TEMPLATE_CODE})

set(GEN_FILES ${FUNCTION_SAVE_PATH}/export_functions.cpp)
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=torch
    BYPRODUCTS ${GEN_FILES}
    DEPENDS test_gen_dependency)

set(FUNCTIONS_SRC ${GEN_FILES})

pybind11_add_module(${DIOPIFUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPIFUNCTIONS} PRIVATE diopirt)
add_dependencies(${DIOPIFUNCTIONS} test_code_gen)


file(MAKE_DIRECTORY ${DIOPI_TEST_DIR}/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${DIOPI_TEST_DIR}/python/diopilib
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${DIOPI_TEST_DIR}/python/diopilib
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIRT})
