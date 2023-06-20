include(FindPackageHandleStandardArgs)

find_path(TOPSRT_INCLUDE_DIR
    NAMES tops_runtime.h
    HINTS /usr/include/tops/ /usr/include/ /usr/include/dtu/ /usr/include/dtu/tops
    /usr/include/dtu/3_0/runtime
)

find_library(TOPSRT_LIBRARIES
    NAMES topsrt_wrapper
    HINTS /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
)

find_package_handle_standard_args(TOPSRT DEFAULT_MSG
    TOPSRT_INCLUDE_DIR
    TOPSRT_LIBRARIES)

mark_as_advanced(TOPSRT_INCLUDE_DIR TOPSRT_LIBRARIES)