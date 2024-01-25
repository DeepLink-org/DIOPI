# !/bin/bash
set -e

CURRENT_DIR=$(
  cd $(dirname $0)
  pwd
)
DIOPI_TEST_PATH=${CURRENT_DIR}/../../diopi_test

case $1 in
  tops)
    mkdir -p build && cd build
    # do not use && between cmake and make -j, because the cmake error will not incur error exit (return code is 0) in switch case statementts.
    cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") .. -DIMPL_OPT=tops -DTEST=ON -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTOPS_HOME=/home/cse/src/install/usr
    make -j8
    ;;
  clean)
    rm -rf build
    ;;
  cuda)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=cuda -DTEST=ON
    make -j8
    ;;
  torch)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=torch -DCMAKE_BUILD_TYPE=Release -DTEST=ON \
      -DENABLE_COVERAGE=${USE_COVERAGE} -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
    make -j8
    ;;
  torch_dyload)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=torch -DCMAKE_BUILD_TYPE=Release -DDYLOAD=ON -DTEST=ON \
      -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') && make -j8
    mkdir -p ${DIOPI_TEST_PATH}/lib && ln -sf ${CURRENT_DIR}/../lib/libdiopi_real_impl.so ${DIOPI_TEST_PATH}/lib
    ;;
  camb_pytorch)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=camb_pytorch -DCMAKE_BUILD_TYPE=Release -DTEST=ON \
      -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
    make -j8
    ;;
  camb)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=camb -DCMAKE_BUILD_TYPE=Release -DTEST=ON -DENABLE_COVERAGE=${USE_COVERAGE}
    make -j8
    ;;
  ascend)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=ascend -DCMAKE_BUILD_TYPE=Release -DTEST=ON -DENABLE_COVERAGE=${USE_COVERAGE}
    make -j16
    ;;
  hip_pytorch)
    mkdir build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=TORCH -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DHIP=ON
    make -j8
    ;;
  mmcv_ext)
    cd third_party/mmcv_diopi && rm -rf build && mkdir build
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    ;;
  droplet)
    mkdir build && cd build
    cmake .. -DIMPL_OPT=droplet -DTEST=ON -DCMAKE_BUILD_TYPE=Release
    make -j8
    ;;
  supa)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=supa -DCMAKE_BUILD_TYPE=Release -DTEST=ON
    make -j8
    ;;
  kunlunxin)
    mkdir -p build && cd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DIMPL_OPT=kunlunxin -DCMAKE_BUILD_TYPE=Release -DTEST=ON
    make -j32
    ;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1 || exit 1
    ;;
esac
