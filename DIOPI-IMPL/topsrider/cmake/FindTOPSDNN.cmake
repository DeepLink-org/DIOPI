include(FindPackageHandleStandardArgs)

find_path(TOPSDNN_INCLUDE_DIR
    NAMES topsdnn.h
    HINTS /usr/include
    /usr/local/include
    /usr/include/dtu/topsdnn
)

find_library(TOPSDNN_LIBRARIES
    NAMES topsdnn
    HINTS /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
    /usr/local/topsdnn
    /usr/local/topsdnn/lib64
    /usr/local/topsdnn/lib
)

if (TOPSDNN_INCLUDE_DIR)
    if(EXISTS ${TOPSDNN_INCLUDE_DIR}/topsdnn_version.h)
        file(READ ${TOPSDNN_INCLUDE_DIR}/topsdnn_version.h TOPSDNN_VERSION_FILE_CONTENTS)
    else()
        file(READ ${TOPSDNN_INCLUDE_DIR}/topsdnn.h TOPSDNN_VERSION_FILE_CONTENTS)
    endif()
    string(REGEX MATCH "define TOPSDNN_MAJOR * +([0-9]+)"
        TOPSDNN_VERSION_MAJOR "${TOPSDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define TOPSDNN_MAJOR * +([0-9]+)" "\\1"
        TOPSDNN_VERSION_MAJOR "${TOPSDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define TOPSDNN_MINOR * +([0-9]+)"
        TOPSDNN_VERSION_MINOR "${TOPSDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define TOPSDNN_MINOR * +([0-9]+)" "\\1"
        TOPSDNN_VERSION_MINOR "${TOPSDNN_VERSION_MINOR}")
    string(REGEX MATCH "define TOPSDNN_PATCHLEVEL * +([0-9]+)"
        TOPSDNN_VERSION_PATCH "${TOPSDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define TOPSDNN_PATCHLEVEL * +([0-9]+)" "\\1"
        TOPSDNN_VERSION_PATCH "${TOPSDNN_VERSION_PATCH}")
endif(TOPSDNN_INCLUDE_DIR)

if(NOT TOPSDNN_VERSION_MAJOR)
    set(TOPSDNN_VERSION "???")
else()
    set(TOPSDNN_VERSION
        "${TOPSDNN_VERSION_MAJOR}.${TOPSDNN_VERSION_MINOR}.${TOPSDNN_VERSION_PATCH}")
endif()

message("TOPSDNN_VERSION:" ${TOPSDNN_VERSION})

find_package_handle_standard_args(TOPSDNN DEFAULT_MSG
    TOPSDNN_INCLUDE_DIR
    TOPSDNN_LIBRARIES)

mark_as_advanced(TOPSDNN_INCLUDE_DIR TOPSDNN_LIBRARIES)