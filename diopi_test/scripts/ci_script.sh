# !/bin/bash
set -e

CURRENT_DIR=$(dirname $0)

DIOPI_TEST_DIR=$(readlink -f $CURRENT_DIR/../)

case $1 in
  py-lint)
    echo "py-lint" && flake8 $DIOPI_TEST_DIR
    ;;
  *)
    echo -e "[ERROR] Incorrect option:" $1 || exit 1
    ;;

esac
