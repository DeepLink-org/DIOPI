# !/bin/bash
set -e

case $1 in
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    # --repository=.. will be deleted when repository changed.
    (echo "cpp-lint" && python scripts/cpplint.py --linelength=160 --repository=.. \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --recursive ./) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;

esac
exit 0