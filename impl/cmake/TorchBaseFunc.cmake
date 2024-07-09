
macro(diopi_find_torch)
  execute_process(
    COMMAND sh -c "python -c 'import torch;print(torch.utils.cmake_prefix_path)'"
    OUTPUT_VARIABLE DIOPI_TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "DIOPI_TORCH_CMAKE_PREFIX:${DIOPI_TORCH_CMAKE_PREFIX}")
  if(DIOPI_TORCH_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH ${DIOPI_TORCH_CMAKE_PREFIX})
  endif()

  find_package(Torch REQUIRED)
  if (Torch_FOUND)
      message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
      message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
  
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
      add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
      add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
      add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})
      add_definitions(-DTORCH_VERSION=${Torch_VERSION})
      message(STATUS "Found Torch Version: ${Torch_VERSION}")
  endif()

endmacro()
