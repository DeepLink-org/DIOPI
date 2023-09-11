set(DIOPIRT export_runtime)
set(DIOPIFUNCTIONS export_functions)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/../diopi_test/diopi_stub/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

set(FUNCTION_SAVE_PATH "${CMAKE_SOURCE_DIR}/../diopi_test/diopi_stub/csrc")
set(TEST_GEN_PATH "${CMAKE_SOURCE_DIR}/../diopi_test/diopi_stub/codegen")

set(RUNTIME_SRC
    ${FUNCTION_SAVE_PATH}/litert.cpp
    conform_test.cpp
)
set(EXPORT_SRC
    ${FUNCTION_SAVE_PATH}/export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(${DIOPIRT} PRIVATE diopirt)
target_link_libraries(diopirt ${DEVICEIMPL})

add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=ascend)
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


file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIRT})
