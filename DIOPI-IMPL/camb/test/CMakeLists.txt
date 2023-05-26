# set(DIOPIRT diopirt)

# include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")

# set(TEST_SRC
#     litert.cpp
#     conform_test.cpp)
# message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

# add_library(${DIOPIRT} SHARED ${TEST_SRC})
# set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")
# target_link_libraries(${DIOPIRT} cnnl cnmlrt)

set(DIOPIRT diopi_runtime)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")
include_directories(${PYTHON_INCLUDE_DIR})

set(RUNTIME_SRC
    litert.cpp
    conform_test.cpp
)

set(EXPORT_SRC
    export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(diopirt cnnl cnmlrt)
target_link_libraries(${DIOPIRT} PRIVATE diopirt)
target_link_libraries(diopirt ${DEVICEIMPL})


set(TEST_GEN_PATH "${CMAKE_SOURCE_DIR}/../DIOPI-TEST/codegen")
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=torch --use_adaptor=USE_ADAPTOR)

set(FUNCTION_SAVE_PATH "${CMAKE_SOURCE_DIR}/../DIOPI-TEST/csrc")
file(MAKE_DIRECTORY ${FUNCTION_SAVE_PATH})
file(TOUCH ${FUNCTION_SAVE_PATH}/diopi_functions.cpp)
set(FUNCTIONS_SRC
    diopi_functions.cpp
)


pybind11_add_module(diopi_functions SHARED ${FUNCTIONS_SRC})
target_link_libraries(diopi_functions PRIVATE diopirt)
add_dependencies(diopi_functions test_code_gen)
add_dependencies(diopi_functions adaptor_code_gen)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:diopi_functions> ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python
    DEPENDS diopi_functions ${DIOPIRT})
