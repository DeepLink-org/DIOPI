PLATFORM=/mnt/cache/share/platform
ENV_NAME=pt2.0_diopi
export PATH=`python ${PLATFORM}/env/clear_path.py PATH`
export LD_LIBRARY_PATH=`python ${PLATFORM}/env/clear_path.py LD_LIBRARY_PATH`
GCC_ROOT=${PLATFORM}/dep/gcc-7.5
CONDA_ROOT=${PLATFORM}/env/miniconda3.8
export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

export CUDA_PATH=${PLATFORM}/dep/cuda11.2-cudnn8.5
export MPI_ROOT=${PLATFORM}/dep/openmpi-4.0.5-cuda11.2
export NCCL_ROOT=${PLATFORM}/dep/nccl-2.9.8-cuda11.2
export GTEST_ROOT=${PLATFORM}/dep/googletest-gcc5.4


export LD_LIBRARY_PATH=${CONDA_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PLATFORM}/dep/binutils-2.27/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib/:$LD_LIBRARY_PATH
export PIP_CONFIG_FILE=${CONDA_ROOT}/envs/${ENV_NAME}/.pip/pip.conf

export DIOPI_ROOT=$(pwd)/../impl/lib/
export DIPU_ROOT=$(pwd)/DIPU/dipu/torch_dipu
export DIOPI_PATH=$(pwd)/../proto
export DIPU_PATH=${DIPU_ROOT}
export PYTORCH_DIR=${PLATFORM}/env/miniconda3.8/envs/pt2.0_diopi/lib/python3.8/site-packages
export LIBRARY_PATH=$DIPU_ROOT:${DIOPI_ROOT}:${LIBRARY_PATH}; LD_LIBRARY_PATH=$DIPU_ROOT:$DIOPI_ROOT:$LD_LIBRARY_PATH
export PYTHONPATH=${PYTORCH_DIR}:${PYTHONPATH}
export PATH=${GCC_ROOT}/bin:${CONDA_ROOT}/envs/dipu_poc/bin:${CONDA_ROOT}/bin:${PLATFORM}/dep/binutils-2.27/bin:${PATH}
export LD_PRELOAD=${GCC_ROOT}/lib64/libstdc++.so.6
export PYTORCH_TEST_DIR=${PLATFORM}/env/miniconda3.8/envs/pt2.0_diopi/pytorch2.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export NCCL_INCLUDE_DIRS=${NCCL_ROOT}/include
export VENDOR_INCLUDE_DIRS=${CUDA_PATH}/include

#export CUDA_LAUNCH_BLOCKING=1
#export DIPU_FORCE_FALLBACK_OPS_LIST=_index_put_impl_,index.Tensor_out
#export DIPU_DUMP_OP_ARGS=2
#export DIPU_DEBUG_ALLOCATOR=15
export DIPU_CUDA_EVENT_TIMING=1
export DIPU_DEVICE_MEMCACHING_ALGORITHM=BF
export DIPU_HOST_MEMCACHING_ALGORITHM=BF
export DIPU_PATCH_CUDA_CACHED_ALLOCATOR=0
export DIPU_CHECK_TENSOR_DEVICE=1

source activate $ENV_NAME
