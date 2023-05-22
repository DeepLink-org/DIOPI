set(DIOPIRT diopi_runtime)
find_package(CUDA REQUIRED)

add_compile_options(-fno-elide-constructors)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")

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


if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(ADAPTOR_GEN_PATH "${CMAKE_SOURCE_DIR}/../DIOPI-ADAPTOR/codegen")
    add_custom_target(adaptor_code_gen COMMAND python3 ${ADAPTOR_GEN_PATH}/gen.py
        --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${PROJECT_SOURCE_DIR}/test/include
        --config_device=torch)
    set(USE_ADAPTOR "true")
    add_dependencies(diopi_functions adaptor_code_gen)
    add_definitions(-DTEST_USE_ADAPTOR)
else()
    set(USE_ADAPTOR "false")
endif()

add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:diopi_functions> ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python
    DEPENDS diopi_functions ${DIOPIRT})