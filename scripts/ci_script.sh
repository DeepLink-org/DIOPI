# !/bin/bash
set -e

current_path=$(cd "$(dirname "$0")"; pwd)
echo $current_path

case $1 in
  impl-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    # --repository=.. will be deleted when repository changed.
    (echo "cpp-lint" && python scripts/cpplint.py --linelength=160 --repository=.. \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --exclude=${current_path}/../impl/third_party --exclude=${current_path}/../proto --recursive ./ )  \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;;
  proto-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    # --repository=.. will be deleted when repository changed.
    (echo "proto-lint" python scripts/cpplint.py --linelength=240 --filter=-build/header_guard --recursive proto/ ) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;;
  py-lint)
    (echo "py-lint" && flake8 --ignore=E501,F841,W503 diopi_test/python/conformance/diopi_functions.py \
       && flake8 --ignore=E501,F401 --exclude=diopi_test/python/conformance/diopi_functions.py,scripts/cpplint.py,diopi_test/third_party/,diopi_test/python/conformance/model_config/ \
       && flake8 --ignore=E501,F401 diopi_test/python/conformance/model_config/process_config.py diopi_test/python/conformance/model_config/__init__.py ) \
    || exit -1;;
  test-cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    (echo "test-cpp-lint" && python scripts/cpplint.py --linelength=160 \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --exclude=${current_path}/../impl/third_party ) \
    || exit -1;;
    *)
    echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0