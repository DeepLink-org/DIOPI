# !/bin/bash

set -e

CURRENT_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
IMPL_PATH=$(readlink -f ${CURRENT_PATH}/..)

CMAKE_EXPORT_COMPILE_COMMANDS_FILE=${IMPL_PATH}/build/compile_commands.json

case $1 in
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    # --repository=.. will be deleted when repository changed.
    echo "cpp-lint"
    python scripts/cpplint.py --linelength=160 --repository=.. \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --exclude=${IMPL_PATH}/third_party --exclude=${IMPL_PATH}/ascend_npu/third_party --exclude=${IMPL_PATH}/build \
      --recursive ./
    ;;
  clang-tidy)
    if [ -e ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE} ]; then
      python3 ${IMPL_PATH}/../run-clang-tidy.py -p $(dirname "${CMAKE_EXPORT_COMPILE_COMMANDS_FILE}")
    else
      echo "error: compile_commands.json not found." && exit 1
    fi
    ;;
  clang-tidy-ascend)
    if [ -e ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE} ]; then
      python3 ${IMPL_PATH}/../run-clang-tidy.py 'impl/ascend(?!_npu)' 'impl/ascend_npu/diopi_impl' -p $(dirname ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE})
    else
      echo "error: compile_commands.json not found." && exit 1
    fi
    ;;
  *)
    echo -e "[ERROR] Incorrect option:" $1 && exit 1
    ;;
esac
