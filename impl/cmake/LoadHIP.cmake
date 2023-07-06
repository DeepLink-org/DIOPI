# ROCM_PATH
IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()

#HIP_PATH
set(HIP_PATH ${ROCM_PATH}/hip)

# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake ${CMAKE_PREFIX_PATH})

# Find the HIP Package
set(USE_HIP OFF)
FIND_PACKAGE(HIP 1.0 REQUIRED)
if(HIP_FOUND)
  set(USE_HIP ON)
endif()

set(PARROTS_HIP_TARGETS " --amdgpu-target=gfx900 --amdgpu-target=gfx906 ")
list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS})
if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
    list(APPEND HIP_HIPCC_FLAGS  ${CMAKE_CXX_FLAGS_MINSIZEREL})
endif()

find_package(rocblas PATHS  ${ROCM_PATH}/rocblas/lib)
find_package(hiprand PATHS  ${ROCM_PATH}/hiprand/lib)
find_package(rocfft PATHS  ${ROCM_PATH}/rocfft/lib)
find_package(hipsparse PATHS  ${ROCM_PATH}/hipsparse/lib)
