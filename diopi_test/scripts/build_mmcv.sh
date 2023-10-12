# !/bin/bash
set -e

current_path=$(cd "$(dirname "$0")"; pwd)

case $1 in
  mmcv_ext)
    (mkdir third_party && cd ${current_path}/../third_party && rm -rf mmcv\
        && /mnt/lustre/share/git clone https://github.com/open-mmlab/mmcv.git && cd mmcv\
        && ls && conda list && rm -rf build && mkdir build \
        && MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i) \
    || exit -1;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1;

esac
exit 0
