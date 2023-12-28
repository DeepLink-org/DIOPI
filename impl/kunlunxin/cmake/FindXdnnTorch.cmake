## xdnn_pytorch
find_path(XDNNTORCH_INCLUDE_DIR
    NAMES xdnn_pytorch/xdnn_pytorch.h
    HINTS ${XDNNTORCH_TOOLKIT_ROOT}/include
          $ENV{XDNNTORCH_TOOLKIT_ROOT}/include
)
message("XDNNTORCH_INCLUDE_DIR:" ${XDNNTORCH_INCLUDE_DIR})
find_library(XDNNTORCH_LIBRARIES
    NAMES libxdnn_pytorch.so
    HINTS ${XDNNTORCH_TOOLKIT_ROOT}/so
          $ENV{XDNNTORCH_TOOLKIT_ROOT}/so
)
message("XDNNTORCH_LIBRARIES:" ${XDNNTORCH_LIBRARIES})
if(NOT XDNNTORCH_INCLUDE_DIR OR NOT XDNNTORCH_LIBRARIES)
    message(FATAL_ERROR "Cannot find XdnnTorch TOOLKIT for kunlunxin, set ENV 'XDNNTORCH_TOOLKIT_ROOT' correctly")
endif()

## xdnn
find_path(XDNN_INCLUDE_DIR
    NAMES xpu/xdnn.h
    HINTS ${XDNN_TOOLKIT_ROOT}/include
          $ENV{XDNN_TOOLKIT_ROOT}/include
)
message("XDNN_INCLUDE_DIR:" ${XDNN_INCLUDE_DIR})
find_library(XDNN_LIBRARIES
    NAMES libxpuapi.so
    HINTS ${XDNN_TOOLKIT_ROOT}/so
          $ENV{XDNN_TOOLKIT_ROOT}/so
)
message("XDNN_LIBRARIES:" ${XDNN_LIBRARIES})
if(NOT XDNN_INCLUDE_DIR OR NOT XDNN_LIBRARIES)
    message(FATAL_ERROR "Cannot find Xdnn TOOLKIT for kunlunxin, set ENV 'XDNN_TOOLKIT_ROOT' correctly")
endif()

## runtime
find_path(XPURT_INCLUDE_DIR
    NAMES xpu/runtime.h
    HINTS ${XPURT_TOOLKIT_ROOT}/include
          $ENV{XPURT_TOOLKIT_ROOT}/include
)
message("XPURT_INCLUDE_DIR:" ${XPURT_INCLUDE_DIR})
find_library(XPURT_LIBRARIES
    NAMES libxpurt.so
    HINTS ${XPURT_TOOLKIT_ROOT}/so
          $ENV{XPURT_TOOLKIT_ROOT}/so
)
message("XPURT_LIBRARIES:" ${XPURT_LIBRARIES})
if(NOT XPURT_INCLUDE_DIR OR NOT XPURT_LIBRARIES)
    message(FATAL_ERROR "Cannot find XPURT TOOLKIT for kunlunxin, set ENV 'XPURT_TOOLKIT_ROOT' correctly")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XdnnTorch DEFAULT_MSG
    XDNNTORCH_INCLUDE_DIR
    XDNNTORCH_LIBRARIES
    XDNN_INCLUDE_DIR
    XDNN_LIBRARIES
    XPURT_INCLUDE_DIR
    XPURT_LIBRARIES)

mark_as_advanced(XDNNTORCH_INCLUDE_DIR XDNNTORCH_LIBRARIES XDNN_INCLUDE_DIR XDNN_LIBRARIES)
