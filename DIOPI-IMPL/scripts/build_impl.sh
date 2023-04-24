# !/bin/bash
set -e

CURRENT_DIR=$(cd $(dirname $0); pwd)
DIOPI_TEST_PATH=${CURRENT_DIR}/../../DIOPI-TEST

echo ${DIOPI_BUILD_TESTRT}

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
    && make && mkdir -p ${DIOPI_TEST_PATH}/lib && ln -sf ${CURRENT_DIR}/../lib/libdiopi_real_impl.so ${DIOPI_TEST_PATH}/lib;;
  camb_pytorch)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=camb_pytorch -DTEST=${DIOPI_BUILD_TESTRT} \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    && make;;
  camb)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=camb -DTEST=${DIOPI_BUILD_TESTRT} && make;;
  ascend)
    mkdir -p build && cd build && cmake .. -DIMPL_OPT=ascend -DTEST=${DIOPI_BUILD_TESTRT} && make;;
  hip_pytorch)
    mkdir build && cd build && cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIMPL_OPT=TORCH -DHIP=ON && make -j4 \
    || exit -1;;
  mmcv_ext)
    (cd third_party/mmcv_diopi && rm -rf build && mkdir build \
        && MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i) \
    || exit -1;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;
esac

exit 0
