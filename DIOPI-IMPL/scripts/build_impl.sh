# !/bin/bash
set -e

CURRENT_DIR=$(cd $(dirname $0); pwd)
DIOPI_TEST_PATH=${CURRENT_DIR}/../../DIOPI-TEST

case $1 in
  cuda)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=cuda && make;;
  torch)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=torch -DDEBUG=ON\
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  torch_test)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=torch -DDEBUG=ON -DTEST=ON\
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  torch_dyload)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=torch -DDEBUG=ON -DDYLOAD=ON\
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  camb_pytorch)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=camb_pytorch \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  camb)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=camb && make;;
  ascend)
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=ascend && make;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;
esac

mkdir -p ${DIOPI_TEST_PATH}/lib
ln -sf ${CURRENT_DIR}/../lib/libdiopi_impl.so ${DIOPI_TEST_PATH}/lib

exit 0