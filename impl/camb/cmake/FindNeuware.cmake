find_path(NEUWARE_ROOT_DIR
    NAMES include/cnrt.h
    HINTS ${NEUWARE_ROOT}
          $ENV{NEUWARE_ROOT}
          ${CAMB_PATH}
          $ENV{CAMB_PATH}
          /usr/local/neuware
)

find_program(NEUWARE_CNCC_EXECUTABLE
    NAMES cncc
    HINTS ${NEUWARE_ROOT_DIR}/bin
          ${NEUWARE_ROOT}/bin
          $ENV{NEUWARE_ROOT}/bin
          ${CAMB_PATH}/bin
          $ENV{CAMB_PATH}/bin
          /usr/local/neuware/bin
)


if(NOT NEUWARE_ROOT_DIR)
    message(FATAL_ERROR "Cannot find NEUWARE SDK for cambricon, set ENV 'NEUWARE_ROOT' correctly")
endif()

# get CNRT version
file(READ ${NEUWARE_ROOT_DIR}/include/cnrt.h CNRT_FILE_CONTENTS)
string(REGEX MATCH "define CNRT_MAJOR_VERSION * +([0-9]+)"
    MATCH_RESULT "${CNRT_FILE_CONTENTS}")
set(CNRT_MAJOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define CNRT_MINOR_VERSION * +([0-9]+)"
    MATCH_RESULT "${CNRT_FILE_CONTENTS}")
set(CNRT_MINOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define CNRT_PATCH_VERSION * +([0-9]+)"
    MATCH_RESULT "${CNRT_FILE_CONTENTS}")
set(CNRT_PATCH_VERSION "${CMAKE_MATCH_1}")
set(CNRT_VERSION "${CNRT_MAJOR_VERSION}.${CNRT_MINOR_VERSION}.${CNRT_PATCH_VERSION}")

# get CNNL version
file(READ ${NEUWARE_ROOT_DIR}/include/cnnl.h CNNL_FILE_CONTENTS)
string(REGEX MATCH "define CNNL_MAJOR * +([0-9]+)"
    MATCH_RESULT "${CNNL_FILE_CONTENTS}")
set(CNNL_MAJOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define CNNL_MINOR * +([0-9]+)"
    MATCH_RESULT "${CNNL_FILE_CONTENTS}")
set(CNNL_MINOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define CNNL_PATCHLEVEL * +([0-9]+)"
    MATCH_RESULT "${CNNL_FILE_CONTENTS}")
set(CNNL_PATCH_VERSION "${CMAKE_MATCH_1}")
set(CNNL_VERSION "${CNNL_MAJOR_VERSION}.${CNNL_MINOR_VERSION}.${CNNL_PATCH_VERSION}")

set(CMAKE_CNCC_FLAGS "-I${NEUWARE_ROOT_DIR}/include" "--std=c++11")
if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND CMAKE_CNCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Neuware
  REQUIRED_VARS
    NEUWARE_ROOT_DIR
    NEUWARE_CNCC_EXECUTABLE
    CMAKE_CNCC_FLAGS
  VERSION_VAR
    CNRT_VERSION
)

# setup flags for C++ compile
include_directories(SYSTEM "${NEUWARE_ROOT_DIR}/include")
link_directories("${NEUWARE_ROOT_DIR}/lib64")
