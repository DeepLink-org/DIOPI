find_library(
  DIOPI_TORCH_EXT_FLASH_ATTN_LIB
  NAMES # Set env FLASH_ATTN_LIB_NAME if your lib has a different name
        ENV FLASH_ATTN_LIB_NAME
        # this is the default name of the flash-attention library
        flash_attn_2_cuda.cpython-38-x86_64-linux-gnu.so
  HINTS
    # Set environment variable FLASH_ATTN_LIB_DIR to the path of the library
    ENV FLASH_ATTN_LIB_DIR
    # This is the path on cluster 1424 A100
    /mnt/cache/share/platform/dep/DIOPI_pytorch/flash-attention/build/lib.linux-x86_64-3.10
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
