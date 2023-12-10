# !/bin/bash
set -e

CURRENT_DIR=$(dirname $0)

DIOPI_TEST_DIR=$(readlink -f $CURRENT_DIR/../)

case $1 in
  py-lint)
    echo "py-lint" && flake8 $DIOPI_TEST_DIR
    ;;
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    echo "cpp-lint"
    python scripts/cpplint.py --exclude=impl/third_party/ --linelength=160 --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --recursive $DIOPI_TEST_DIR/diopi_stub/csrc $DIOPI_TEST_DIR/diopi_stub/include
    ;;
  *)
    echo -e "[ERROR] Incorrect option:" $1 || exit 1
    ;;

esac
