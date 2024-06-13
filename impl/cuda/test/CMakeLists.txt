set(DIOPIRT diopirt)
find_package(CUDA REQUIRED)

set(DIOPIRT export_runtime)
set(DIOPIFUNCTIONS export_functions)
find_package(CUDA REQUIRED)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${DIOPI_IMPL_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

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

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/csrc")

set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/codegen")
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=torch --use_adaptor=${USE_ADAPTOR})
add_custom_target(functions_copy ALL
    COMMAND ln -f ${FUNCTION_SAVE_PATH}/export_functions.cpp ${PROJECT_SOURCE_DIR}/test
    DEPENDS test_code_gen)

file(TOUCH export_functions.cpp)
set(FUNCTIONS_SRC
    export_functions.cpp
)

pybind11_add_module(${DIOPIFUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPIFUNCTIONS} PRIVATE diopirt)
add_dependencies(${DIOPIFUNCTIONS} functions_copy)
if(${USE_ADAPTOR} STREQUAL "true")
    add_dependencies(${DIOPIFUNCTIONS} adaptor_code_gen)
endif()


file(MAKE_DIRECTORY ${DIOPI_TEST_DIR}/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${DIOPI_TEST_DIR}/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${DIOPI_TEST_DIR}/python
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIRT})
