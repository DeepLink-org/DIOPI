set(DIOPIEXPORTRT export_runtime)
set(DIOPIFUNCTIONS export_functions)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

# lib
set(DROPLET_INSTALL "/usr/local/tangrt")
include_directories("${DROPLET_INSTALL}/include")
link_directories("${DROPLET_INSTALL}/lib/linux-x86_64")

find_program(DROPLET_CLANG_EXECUTABLE
    NAMES ptcc
    HINTS ${DROPLET_INSTALL}/bin
)

set(DIOPI_TEST_DIR "${CMAKE_SOURCE_DIR}/../diopi_test")

include_directories(SYSTEM "${DROPLET_INSTALL}/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")
include_directories(SYSTEM $ENV{PYTHON_INCLUDE_DIR})
include_directories(SYSTEM "${DIOPI_TEST_DIR}/diopi_stub/include")
set(RUNTIME_SRC
    litert.cpp
    conform_test.cpp
)

set(EXPORT_SRC
    export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${DIOPIEXPORTRT} SHARED ${EXPORT_SRC})
add_library(diopiruntime SHARED ${RUNTIME_SRC})

if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../adaptor/codegen")
    add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
        --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${CMAKE_SOURCE_DIR}/../adaptor
        --config_device=droplet)
    set(USE_ADAPTOR "true")
    add_dependencies(diopiruntime adaptor_code_gen)
    add_definitions(-DTEST_USE_ADAPTOR)
else()
    set(USE_ADAPTOR "false")
endif()

target_link_libraries(${DIOPIEXPORTRT} PRIVATE diopiruntime)
target_link_libraries(diopiruntime tangrt_shared)
target_link_libraries(diopiruntime ${DIOPI_DROPLET_LIB})

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/diopi_stub/csrc")

set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/diopi_stub/codegen")
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=droplet)
add_custom_target(functions_copy ALL
    COMMAND ln -f ${FUNCTION_SAVE_PATH}/export_functions.cpp ${PROJECT_SOURCE_DIR}/test
    DEPENDS test_code_gen)

# file(MAKE_DIRECTORY ${FUNCTION_SAVE_PATH})
file(TOUCH export_functions.cpp)
set(FUNCTIONS_SRC
    export_functions.cpp
)

pybind11_add_module(${DIOPIFUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPIFUNCTIONS} PRIVATE diopiruntime)
add_dependencies(${DIOPIFUNCTIONS} functions_copy)
if(${USE_ADAPTOR} STREQUAL "true")
    add_dependencies(${DIOPIFUNCTIONS} adaptor_code_gen)
endif()


file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${CMAKE_SOURCE_DIR}/../diopi_test/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIEXPORTRT}> ${CMAKE_SOURCE_DIR}/../diopi_test/python
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIEXPORTRT})
