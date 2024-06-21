
macro(InitFindTorch)
  execute_process(
    COMMAND sh -c "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
    OUTPUT_VARIABLE DIOPI_TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "DIOPI_TORCH_CMAKE_PREFIX:${DIOPI_TORCH_CMAKE_PREFIX}")
  if(DIOPI_TORCH_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH ${DIOPI_TORCH_CMAKE_PREFIX})
  endif()
endmacro()
