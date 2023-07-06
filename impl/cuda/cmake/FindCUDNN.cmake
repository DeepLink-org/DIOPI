# CMake script to locate cuDNN

include(FindPackageHandleStandardArgs)

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${CUDA_TOOLKIT_CUDNN_ROOT}/include
          $ENV{CUDA_TOOLKIT_CUDNN_ROOT}/include
          ${CUDA_BIN_PATH}/include
          $ENV{CUDA_BIN_PATH}/include
          ${CUDA_PATH}/include
          $ENV{CUDA_PATH}/include
          /usr/include
          /usr/local/include
          /usr/local/cudnn
          /usr/local/cudnn/include
          /usr/local/cuda/include
)

find_library(CUDNN_LIBRARIES
    NAMES cudnn
    HINTS ${CUDA_TOOLKIT_CUDNN_ROOT}/lib64
          ${CUDA_TOOLKIT_CUDNN_ROOT}/lib
          $ENV{CUDA_TOOLKIT_CUDNN_ROOT}/lib64
          $ENV{CUDA_TOOLKIT_CUDNN_ROOT}/lib
          ${CUDA_BIN_PATH}/lib64
          ${CUDA_BIN_PATH}/lib
          $ENV{CUDA_BIN_PATH}/lib64
          $ENV{CUDA_BIN_PATH}/lib
          ${CUDA_PATH}/lib64
          ${CUDA_PATH}/lib
          $ENV{CUDA_PATH}/lib64
          $ENV{CUDA_PATH}/lib
          /usr/lib64
          /usr/lib
          /usr/local/lib64
          /usr/local/lib
          /usr/local/cudnn
          /usr/local/cudnn/lib64
          /usr/local/cudnn/lib
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
)

if (CUDNN_INCLUDE_DIR)
    if(EXISTS ${CUDNN_INCLUDE_DIR}/cudnn_version.h)
        # cudnn8 change verion info from cudnn.h to cudnn_version.h
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn_version.h CUDNN_VERSION_FILE_CONTENTS)
    else()
        # read cudnn verion
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
    endif()
    # cudnn v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
        CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
        CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
endif(CUDNN_INCLUDE_DIR)

if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "???")
else()
    set(CUDNN_VERSION
        "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
endif()

find_package_handle_standard_args(CUDNN DEFAULT_MSG
    CUDNN_INCLUDE_DIR
    CUDNN_LIBRARIES)

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARIES)
