add_library(flash_attn_2_cuda.cpython-38-x86_64-linux-gnu SHARED IMPORTED)
set_target_properties(flash_attn_2_cuda.cpython-38-x86_64-linux-gnu PROPERTIES
  IMPORTED_LOCATION /mnt/cachenew/lilingjie/workspace/flash-attention/build/lib.linux-x86_64-cpython-38/flash_attn_2_cuda.cpython-38-x86_64-linux-gnu.so
  PREFIX "")
  
add_library(diopi_torch_ext_flash_attn INTERFACE)
target_link_libraries(diopi_torch_ext_flash_attn INTERFACE flash_attn_2_cuda.cpython-38-x86_64-linux-gnu)
target_include_directories(diopi_torch_ext_flash_attn SYSTEM
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )
