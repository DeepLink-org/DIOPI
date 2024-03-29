# !/bin/bash
set -e

current_path=$(
  cd "$(dirname "$0")"
  pwd
)

case $1 in
  mmcv_ext)
    cd ${current_path}/../third_party/mmcv_diopi
    rm -rf build && mkdir build
    MMCV_WITH_DIOPI=1 MMCV_WITH_OPS=1 python setup.py build_ext -i
    ;;
  *)
    echo -e "[ERROR] Incorrect compilation option:" $1 && exit 1
    ;;

esac
