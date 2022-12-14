# !/bin/bash
set -e

case $1 in
  cuda)
    (rm -rf build && mkdir build && cd build \
        && cmake .. -DCUDA_ARCH_AUTO=ON -DIMPL_OPT=CUDA && make -j4) \
    || exit -1;;
  torch)
    (rm -rf build && mkdir build && cd build \
      && cmake .. -DIMPL_OPT=TORCH \
      -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
      && make -j4) \
    || exit -1;;
  torch_no_runtime)
    (rm -rf build && mkdir build && cd build \
      && cmake .. -DIMPL_OPT=TORCH -DRUNTIME=OFF \
      -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
      && make -j4) \
    || exit -1;;
  camb_pytorch)
    (rm -rf build && mkdir build && cd build \
        && cmake .. -DIMPL_OPT=camb_pytorch && make -j4) \
    || exit -1;;
  camb_pytorch_no_runtime)
    (rm -rf build && mkdir build && cd build \
        && cmake .. -DIMPL_OPT=camb_pytorch -DRUNTIME=OFF && make -j4) \
    || exit -1;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;

esac
exit 0
