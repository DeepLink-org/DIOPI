message(STATUS "flash-attention DISABLED")
add_library(diopi_torch_ext_flash_attn INTERFACE)

target_include_directories(diopi_torch_ext_flash_attn INTERFACE  
   ${BASE_TORCH_DIR}/functions/functions_ext/flash-attention/include)
