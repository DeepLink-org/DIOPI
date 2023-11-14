find_package(PythonInterp REQUIRED)

if (PYTHONINTERP_FOUND)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print(sys.version_info.major)"
        OUTPUT_VARIABLE PYTHON_VERSION_MAJOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print(sys.version_info.minor)"
        OUTPUT_VARIABLE PYTHON_VERSION_MINOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    message(STATUS "Python version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
else()
    message(FATAL_ERROR "Python interpreter not found.")
endif()

string(CONCAT FLASH_ATTN_LIB_DEFAULT_NAME
      "flash_attn_2_cuda.cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}-x86_64-linux-gnu.so")

string(CONCAT FLASH_ATTN_LIB_DEFAULT_DIR
       "/mnt/cache/shenliancheng/workspace/flash-attention/build/lib.linux-x86_64-cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")

find_library(
  DIOPI_TORCH_EXT_FLASH_ATTN_LIB
  NAMES # Set env FLASH_ATTN_LIB_NAME if your lib has a different name
        ENV FLASH_ATTN_LIB_NAME
        # this is the default name of the flash-attention library
        ${FLASH_ATTN_LIB_DEFAULT_NAME}
  HINTS
    # Set environment variable FLASH_ATTN_LIB_DIR to the path of the library
    ENV FLASH_ATTN_LIB_DIR
    # This is the path on cluster 1424 A100
    ${FLASH_ATTN_LIB_DEFAULT_DIR}
)

if(NOT DIOPI_TORCH_EXT_FLASH_ATTN_LIB)
  message(WARNING "flash-attention NOT FOUND, will build without mha support")
  add_library(diopi_torch_ext_flash_attn INTERFACE)
else()
  message(STATUS "FOUND flash-attention: ${DIOPI_TORCH_EXT_FLASH_ATTN_LIB}")
  add_library(diopi_torch_ext_flash_attn SHARED IMPORTED GLOBAL)
  set_target_properties(
    diopi_torch_ext_flash_attn PROPERTIES IMPORTED_LOCATION
                                          ${DIOPI_TORCH_EXT_FLASH_ATTN_LIB})
  target_link_options(diopi_torch_ext_flash_attn INTERFACE "LINKER:-no-as-needed")
endif()

target_include_directories(diopi_torch_ext_flash_attn INTERFACE include)
