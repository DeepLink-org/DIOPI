set(DIOPIRT diopi_runtime)
set(DIOPIFUNCTIONS diopi_functions)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

set(RUNTIME_SRC
    litert.cpp
    conform_test.cpp
)

set(EXPORT_SRC
    diopi_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(${DIOPIRT} PRIVATE diopirt)
target_link_libraries(diopirt ${DEVICEIMPL})

set(FUNCTION_SAVE_PATH "${CMAKE_SOURCE_DIR}/../diopi_test/diopi_stub/csrc")

set(TEST_GEN_PATH "${CMAKE_SOURCE_DIR}/../diopi_test/diopi_stub/codegen")
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=camb --use_adaptor=${USE_ADAPTOR})
add_custom_target(functions_copy ALL
    COMMAND ln -f ${FUNCTION_SAVE_PATH}/diopi_functions.cpp ${PROJECT_SOURCE_DIR}/test
    DEPENDS test_code_gen)

# file(MAKE_DIRECTORY ${FUNCTION_SAVE_PATH})
file(TOUCH diopi_functions.cpp)
set(FUNCTIONS_SRC diopi_functions.cpp)

pybind11_add_module(${DIOPIFUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPIFUNCTIONS} PRIVATE diopirt)
add_dependencies(${DIOPIFUNCTIONS} functions_copy)
if(${USE_ADAPTOR} STREQUAL "true")
    add_dependencies(${DIOPIFUNCTIONS} adaptor_code_gen)
endif()


file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopi_stub)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIRT})
