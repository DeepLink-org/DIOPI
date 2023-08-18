set(DIOPIRT_PYBIND export_runtime)
set(DIOPIFUNCTIONS_PYBIND export_functions)
set(DIOPIRT diopirt)

add_compile_options(-fno-elide-constructors)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11 build)
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

# config inc/lib path for SUPA runtime.
if(NOT DEFINED ENV{SUPA_PATH})
  set(SUPA_PATH /usr/local/supa)
else()
  set(SUPA_PATH $ENV{SUPA_PATH})
endif()

find_path(SUPA_INCLUDE supa.h
  ${SUPA_PATH}/include
)

if(NOT SUPA_INCLUDE)
  message(FATAL_ERROR " Could not find SUPA header files. Make sure fullstack is installed.")
else()
  message(STATUS " SUPA include path: ${SUPA_INCLUDE}")
endif()
include_directories(SYSTEM ${SUPA_INCLUDE})

find_library(SUPA_RUNTIME_LIB supa-runtime ${SUPA_PATH}/lib ${SUPA_PATH}/lib64)
if(NOT SUPA_RUNTIME_LIB)
  message(FATAL_ERROR " Could not find SUPA runtime lib. Make sure fullstack is installed.")
else()
  message(STATUS " SUPA runtime lib: ${SUPA_RUNTIME_LIB}")
endif()

# config path for Python.h
find_package(Python3 COMPONENTS Interpreter Development)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "No sutiable python library found.")
endif()
message(STATUS "Python lib: ${Python3_LIBRARIES};${Python3_LIBRARY_DIRS}")
message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")
include_directories(SYSTEM "${Python3_INCLUDE_DIRS}")

# set diopi_test path
set(DIOPI_TEST_DIR "${CMAKE_SOURCE_DIR}/../diopi_test")

# setup and create test-rt.
include_directories(SYSTEM "${DIOPI_TEST_DIR}/include")

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/csrc")
set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/codegen")

set(RUNTIME_SRC
    conform_test.cpp
    ${FUNCTION_SAVE_PATH}/litert.cpp)

add_library(${DIOPIRT} SHARED ${RUNTIME_SRC})
target_link_libraries(${DIOPIRT} ${SUPA_RUNTIME_LIB})

# link test-rt with real symbols in diopi_impl.so
target_link_libraries(${DIOPIRT} -Wl,--no-as-needed ${DIOPI_BR_LIB} -Wl,--as-needed)

# create pybind library for test-rt
set(RT_EXPORT_SRC
    ${FUNCTION_SAVE_PATH}/export_runtime.cpp
)
pybind11_add_module(${DIOPIRT_PYBIND} SHARED ${RT_EXPORT_SRC})
target_link_libraries(${DIOPIRT_PYBIND} PRIVATE ${DIOPIRT})

# create pybind library for functions.
add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py --device=torch)

set(EXPORT_FUNCTIONS "${FUNCTION_SAVE_PATH}/export_functions.cpp")
file(TOUCH ${EXPORT_FUNCTIONS})
set(FUNCTIONS_EXPORT_SRC
    ${EXPORT_FUNCTIONS}
)

pybind11_add_module(${DIOPIFUNCTIONS_PYBIND} SHARED ${FUNCTIONS_EXPORT_SRC})
target_link_libraries(${DIOPIFUNCTIONS_PYBIND} PRIVATE ${DIOPIRT} ${DEVICEIMPL})
add_dependencies(${DIOPIFUNCTIONS_PYBIND} test_code_gen)

# copy pybind library into test folder.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../diopi_test/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS_PYBIND}> ${CMAKE_SOURCE_DIR}/../diopi_test/python
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT_PYBIND}> ${CMAKE_SOURCE_DIR}/../diopi_test/python
    DEPENDS ${DIOPIFUNCTIONS_PYBIND} ${DIOPIRT_PYBIND})
