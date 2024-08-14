set(RT_PYBIND export_runtime)
set(FUNC_PYBIND export_functions)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)
set(DIOPI_TEST_DIR "${CMAKE_SOURCE_DIR}/../diopi_test")

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")
include_directories(SYSTEM "${DIOPI_TEST_DIR}/diopi_stub/include")
set(TEST_CSRC_PATH "${DIOPI_TEST_DIR}/diopi_stub/csrc")


# diopt rt test.
set(RUNTIME_SRC
    ${TEST_CSRC_PATH}/litert.cpp
    # use torch cuda runtime
    ${BASE_TORCH_DIR}/test/conform_test.cpp
)
cuda_add_library(diopirt SHARED ${RUNTIME_SRC})
message(STATUS "test diopirt CUDA_LIBRARIES is:" ${CUDA_LIBRARIES})
target_link_libraries(diopirt ${CUDA_LIBRARIES})

# rt test py export 
set(RT_PYBIND_SRC
    ${TEST_CSRC_PATH}/export_runtime.cpp
)
pybind11_add_module(${RT_PYBIND} SHARED ${RT_PYBIND_SRC})
target_link_libraries(${RT_PYBIND} PRIVATE diopirt)

# diopi op func test py export
file(GLOB TEST_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${DIOPI_TEST_DIR}/diopi_stub/codegen/*.py)
add_custom_target(test_gen_dependency DEPENDS ${TEST_TEMPLATE_CODE})
set(GEN_FILES ${TEST_CSRC_PATH}/export_functions.cpp)
set(OP_GEN_PATH "${DIOPI_TEST_DIR}/diopi_stub/codegen")
add_custom_target(test_code_gen ALL
    COMMAND python3 ${OP_GEN_PATH}/gen.py --device=muxi
    BYPRODUCTS ${GEN_FILES}
    DEPENDS test_gen_dependency)

set(FUNCTIONS_SRC ${GEN_FILES})

pybind11_add_module(${FUNC_PYBIND} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${FUNC_PYBIND} PRIVATE diopirt ${DEVICEIMPL})
add_dependencies(${FUNC_PYBIND} test_code_gen)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${FUNC_PYBIND}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${RT_PYBIND}> ${CMAKE_SOURCE_DIR}/../diopi_test/python/diopilib
    DEPENDS ${FUNC_PYBIND} ${RT_PYBIND})
