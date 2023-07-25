include(FindPackageHandleStandardArgs)

find_path(OPS_INCLUDE_DIR
    NAMES ops.h
    HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

find_library(OPS_LIBRARIES
    NAMES ops
    HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/prebuilt
)


find_package_handle_standard_args(OPS DEFAULT_MSG
    OPS_INCLUDE_DIR
    OPS_LIBRARIES)

mark_as_advanced(OPS_INCLUDE_DIR OPS_LIBRARIES)
