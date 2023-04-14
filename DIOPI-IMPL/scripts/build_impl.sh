# !/bin/bash
set -e

CURRENT_DIR=$(cd $(dirname $0); pwd)
DIOPI_TEST_PATH=${CURRENT_DIR}/../../DIOPI-TEST

case $1 in
  clean)
    rm -rf build;;
  cuda)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=cuda -DTEST=${DIOPI_BUILD_TESTRT} && make;;
  torch)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=torch -DDEBUG=ON -DTEST=${DIOPI_BUILD_TESTRT} \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  torch_dyload)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=torch -DDEBUG=ON -DDYLOAD=ON -DTEST=${DIOPI_BUILD_TESTRT} \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  camb_pytorch)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=camb_pytorch -DTEST=${DIOPI_BUILD_TESTRT} \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  camb)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=camb -DTEST=${DIOPI_BUILD_TESTRT} && make;;
  ascend)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=ascend -DTEST=${DIOPI_BUILD_TESTRT} && make;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;
esac

if [[ -z $DIOPI_BUILD_TESTRT ]] && [[ $DIOPI_BUILD_TESTRT = "ON" ]] ; then
  mkdir -p ${DIOPI_TEST_PATH}/lib
  ln -sf ${CURRENT_DIR}/../lib/libdiopi_impl.so ${DIOPI_TEST_PATH}/lib
  ln -sf ${CURRENT_DIR}/../lib/libdiopirt.so ${DIOPI_TEST_PATH}/lib
fi

exit 0
