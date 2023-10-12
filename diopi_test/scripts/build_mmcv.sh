# !/bin/bash
set -e

current_path=$(cd "$(dirname "$0")"; pwd)

case $1 in
  mmcv_ext)
    (cd ${current_path}/../third_party && rm -rf mmcv\
        && git clone https://github.com/open-mmlab/mmcv.git && cd mmcv\
        && git checkout \
        && ls && conda list && rm -rf build && mkdir build \
        && MMCV_WITH_DIOPI=0 MMCV_WITH_OPS=1 python setup.py build_ext -i) \
    || exit -1;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;

esac
exit 0
