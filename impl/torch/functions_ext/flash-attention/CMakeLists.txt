file(GLOB FLASH_ATTN_CXX_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/*.cpp")
file(GLOB FLASH_ATTN_CU_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/src/*.cu")
cuda_add_library(diopi_torch_ext_flash_attn STATIC ${FLASH_ATTN_CXX_SRCS} ${FLASH_ATTN_CU_SRCS})
target_include_directories(diopi_torch_ext_flash_attn SYSTEM
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(diopi_torch_ext_flash_attn ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(diopi_torch_ext_flash_attn PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF)
