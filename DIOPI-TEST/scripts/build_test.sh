# !/bin/bash
set -e

# CURRENT_DIR=$(cd $(dirname $0); pwd)
# DIOPI_PROTO_PATH=${CURRENT_DIR}/../../DIOPI-PROTO
# DIOPI_IMPL_PATH=${CURRENT_DIR}/../../DIOPI-IMPL
# echo ${DIOPI_PROTO_PATH}
# DST_SRC_DIR=${CURRENT_DIR}/../dist/src
# DST_DIR=${CURRENT_DIR}/../dist
# mkdir -p ${DST_SRC_DIR} ${DST_INCLUDE_DIR}

# cp -fa ${DIOPI_PROTO_PATH}/include/ ${DST_DIR}/

case $1 in
  cuda)
    # ln -sf ${DIOPI_IMPL_PATH}/cuda/test/conform_test.cpp ${DST_SRC_DIR}/
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=cuda && make;;
  torch)
    # ln -sf ${DIOPI_IMPL_PATH}/torch/test/conform_test.cpp ${DST_SRC_DIR}/
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=torch -DDEBUG=ON && make;;
  camb_pytorch)
    # ln -sf ${DIOPI_IMPL_PATH}/camb_pytorch/test/conform_test.cpp ${DST_SRC_DIR}/
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=camb_pytorch && make;;
  camb)
    # ln -sf ${DIOPI_IMPL_PATH}/camb/test/conform_test.cpp ${DST_SRC_DIR}/
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=camb && make;;
  ascend)
    # ln -sf ${DIOPI_IMPL_PATH}/ascend/test/conform_test.cpp ${DST_SRC_DIR}/
    rm -rf build && mkdir build && cd build && cmake .. -DIMPL_OPT=ascend && make;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;
esac

exit 0