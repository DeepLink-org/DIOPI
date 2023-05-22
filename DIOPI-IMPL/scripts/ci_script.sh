# !/bin/bash
set -e

CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)
CMAKE_EXPORT_COMPILE_COMMANDS_FILE=${CURRENT_PATH}/../build/compile_commands.json
case $1 in
  cpp-lint)
    # for other cpplint version, maybe  -whitespace/indent is needed to check impl
    # --repository=.. will be deleted when repository changed.
    (echo "cpp-lint" && python scripts/cpplint.py --linelength=160 --repository=.. \
      --filter=-build/c++11,-legal/copyright,-build/include_subdir,-runtime/references,-runtime/printf,-runtime/int,-build/namespace \
      --exclude=${CURRENT_PATH}/../third_party --exclude=${CURRENT_PATH}/../build \
      --recursive ./ )  \
    || exit -1;;
  clang-tidy)
    (echo "CMAKE_EXPORT_COMPILE_COMMANDS_FILE: ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE}"
    if [ -e ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE} ]; then
      echo "111111111111111"
      python3 ${CURRENT_PATH}/../../run-clang-tidy.py -p `dirname "${CMAKE_EXPORT_COMPILE_COMMANDS_FILE}"`
    else
      echo "error: compile_commands.json not found."
      exit 1
    fi);;
  *)
    echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0