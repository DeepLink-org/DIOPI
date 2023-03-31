# !/bin/bash
set -e

current_path=$(cd "$(dirname "$0")"; pwd)
echo $current_path

case $1 in
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    # --repository=.. will be deleted when repository changed.
    (echo "cpp-lint" && python scripts/cpplint.py --linelength=160 --repository=.. \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --exclude=${current_path}/../third_party --recursive ./ )  \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;

esac
exit 0