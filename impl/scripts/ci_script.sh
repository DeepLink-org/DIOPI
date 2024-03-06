# !/bin/bash

set -e



CURRENT_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
IMPL_PATH=$(readlink -f ${CURRENT_PATH}/..)

CMAKE_EXPORT_COMPILE_COMMANDS_FILE=${IMPL_PATH}/build/compile_commands.json

function download_clangd_tidy {
  # Required tools.
  [ -x "$(command -v git)" ] || { echo "::error::Missing git tool" && exit 1; }
  [ -x "$(command -v clangd)" ] || { echo "::error::Missing clangd tool" && exit 1; }
  # download
  [ -d "$CURRENT_PATH/clangd-tidy" ] ||
  git -c advice.detachedHead=false clone --depth 1 -b v0.1.2 https://github.com/lljbash/clangd-tidy.git "$CURRENT_PATH/clangd-tidy"
}


case $1 in
  clang-tidy)
    if [ -e ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE} ]; then
      # #when clangd-tidy can be downloaded, replace clang-tidy with clangd-tidy.
      # download_clangd_tidy
      # find camb ../adaptor/csrc \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) |
      # xargs "$CURRENT_PATH/clangd-tidy/clangd-tidy" -j8 -v -p $(dirname "${CMAKE_EXPORT_COMPILE_COMMANDS_FILE}")
      python3 ${IMPL_PATH}/../run-clang-tidy.py -p $(dirname "${CMAKE_EXPORT_COMPILE_COMMANDS_FILE}")
    else
      echo "error: compile_commands.json not found." && exit 1
    fi
    ;;
  clang-tidy-ascend)
    if [ -e ${CMAKE_EXPORT_COMPILE_COMMANDS_FILE} ]; then
      download_clangd_tidy
      # Collect source files and run tidy.
      find ascend ascend_npu/diopi_impl ../adaptor/csrc \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) |
      xargs "$CURRENT_PATH/clangd-tidy/clangd-tidy" -j8 -v -p $(dirname "${CMAKE_EXPORT_COMPILE_COMMANDS_FILE}")
    else
      echo "error: compile_commands.json not found." && exit 1
    fi
    ;;
  *)
    echo -e "[ERROR] Incorrect option:" $1 && exit 1
    ;;
esac
