set(DIOPIRT diopi_runtime)
find_package(CUDA REQUIRED)

add_compile_options(-fno-elide-constructors)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory(/home/xintian/add_adaptor/senseparrots/3rdparty/DIOPI/DIOPI-TEST/third_party/pybind11 build)
                #  ${PROJECT_BINARY_DIR}/${PARROTS_BINDING}/pybind11)
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/test/include")
include_directories("${PROJECT_SOURCE_DIR}/../../../DIOPI-ADAPTOR/include")
message(${PROJECT_SOURCE_DIR})


set(RUNTIME_SRC
    litert.cpp
    conform_test.cpp
    # diopi_functions.cpp
)

set(EXPORT_SRC
    export_runtime.cpp
)

set(FUNCTIONS_SRC
    # litert.cpp
    # conform_test.cpp
    diopi_functions.cpp
)

# set(PYBIND_SRC
# )

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(diopi_functions SHARED ${FUNCTIONS_SRC})

pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
cuda_add_library(diopirt SHARED ${RUNTIME_SRC})

# set_target_properties(${DIOPIRT} PROPERTIES SUFFIX ".so")
message(${DEVICEIMPL})
target_link_libraries(${DIOPIRT} PRIVATE diopirt)
target_link_libraries(diopirt ${DEVICEIMPL})
target_link_libraries(diopi_functions PRIVATE diopirt)
target_link_libraries(diopi_functions PRIVATE ${DEVICEIMPL})
# target_link_libraries(${DIOPIRT} ${PYTHON_LIBRARIES})
