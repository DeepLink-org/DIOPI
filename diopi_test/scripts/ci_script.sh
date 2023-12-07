# !/bin/bash
set -e

CURRENT_DIR=$(dirname $0)

DIOPI_TEST_DIR=$(readlink -f $CURRENT_DIR/../)

case $1 in
  py-lint)
    echo "py-lint" && flake8 $DIOPI_TEST_DIR || exit -1;;
  *)
    echo -e "[ERROR] Incorrect option:" $1;

esac
exit 0
