add_library(flash_attn_2_cuda.cpython-38-x86_64-linux-gnu SHARED IMPORTED)
set_target_properties(flash_attn_2_cuda.cpython-38-x86_64-linux-gnu PROPERTIES
IMPORTED_LOCATION /mnt/cache/share/platform/dep/DIOPI_pytorch/flash-attention/flash_attn_2_cuda.cpython-38-x86_64-linux-gnu.so
PREFIX "")
  
add_library(diopi_torch_ext_flash_attn INTERFACE)
target_link_libraries(diopi_torch_ext_flash_attn INTERFACE  -Wl,--no-as-needed flash_attn_2_cuda.cpython-38-x86_64-linux-gnu -Wl,--as-needed)
target_include_directories(diopi_torch_ext_flash_attn SYSTEM
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )