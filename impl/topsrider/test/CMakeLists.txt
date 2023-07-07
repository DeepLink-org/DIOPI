set(DIOPIRT export_runtime)
set(DIOPIFUNCTIONS export_functions)

add_compile_options(-fno-elide-constructors -fpermissive)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../../dipu/third_party/DIOPI/DIOPI-IMPL/third_party/pybind11/include")
#include_directories(SYSTEM "/home/cse/miniconda3/envs/dipu/include/python3.8")"
include_directories(SYSTEM "~/miniconda3/envs/dipu/include/python3.8")

set(RUNTIME_SRC
    litert.cpp
    ../conform_test.cpp)

set(EXPORT_SRC
    export_runtime.cpp
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

set(FUNCTION_SAVE_PATH "${CMAKE_SOURCE_DIR}/../DIOPI-TEST/csrc")

set(TEST_GEN_PATH "${CMAKE_SOURCE_DIR}/../DIOPI-TEST/codegen")
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=topsrider --use_adaptor=${USE_ADAPTOR})
add_custom_target(functions_copy ALL
    COMMAND ln -f ${FUNCTION_SAVE_PATH}/export_functions.cpp ${PROJECT_SOURCE_DIR}/test
    DEPENDS test_code_gen)

# file(MAKE_DIRECTORY ${FUNCTION_SAVE_PATH})
file(TOUCH export_functions.cpp)
set(FUNCTIONS_SRC
    export_functions.cpp
)

pybind11_add_module(${DIOPIFUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPIFUNCTIONS} PRIVATE -Wl,--no-as-needed diopirt -Wl,--as-needed)
add_dependencies(${DIOPIFUNCTIONS} functions_copy)
if(${USE_ADAPTOR} STREQUAL "true")
    add_dependencies(${DIOPIFUNCTIONS} adaptor_code_gen)
endif()


file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${CMAKE_SOURCE_DIR}/../DIOPI-TEST/python
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIRT})

