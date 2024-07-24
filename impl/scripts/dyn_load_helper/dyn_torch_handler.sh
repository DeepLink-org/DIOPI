#!/usr/bin/env bash
set -eo pipefail

# pip install patchelf

diopi_suffix=".diopi"
torch_raws=("libtorch.so" "libtorch_cuda.so" "libtorch_cpu.so" "libc10_cuda.so" "libc10.so")
torch_4diopi=()
for ((i=0; i<${#torch_raws[@]}; i++)); do
  torch_4diopi[i]=${torch_raws[$i]}${diopi_suffix}
done

# even using RTLD_DEEPBIND load, same name lib still has only one instance in the address space.
# RTLD_DEEPBIND just make same symbol name in different lib names be loaded as 2 instance but not 
# symbol in same lib name.
function gen_versioned_torch() {
  for ((i=0; i<${#torch_raws[@]}; i++)); do
    cp ${torch_raws[$i]} ${torch_4diopi[$i]}
  done

  for ((i=0; i<${#torch_4diopi[@]}; i++)); do
    libi=${torch_4diopi[$i]}
    replace_items=""
    for ((depIdx=i+1; depIdx<${#torch_4diopi[@]}; depIdx++)); do
      dep_raw=${torch_raws[$depIdx]}
      dep_4diopi=${torch_4diopi[$depIdx]}
      replace_items=${replace_items}" --replace-needed ${dep_raw} ${dep_4diopi}"

    done
     patchelf ${replace_items} --set-soname ${libi} ${libi}
  done
}

function check_correct_torch() {
  echo "check diopi torch: $1"

  # TODO: use an elf lib to remove unqiue flag in *so.diopi
  # remove unique symbols of both cpu torch (dipu use) and device torch (diopi use).
  # if you device torch is compiled by clang, which not supporting -fno-gnu-unique,
  # just test if it works (eg: muxi torch with unique can coexist with no-uniqued cpu torch)

  echo "please check if you torch builded with -fno-gnu-unique to support multi version torch coexist"
  set +e
  chk_ret=`cd $1 && ls -alh |grep .*\.so\.diopi | wc -l`
  set -e
  if [[ ${chk_ret} -ne ${#torch_4diopi[@]} ]]; then
      echo "ret value: ${chk_ret}, in device-torch dir, not find dyn-load needed XX.so.diopi libs!"
      echo "!! please manual run handle_dyload_torch.sh patch_torch {device_torch_dir} to gen dyn-load needed multi-version torch"
      exit -1
  fi

  echo  "diopi torch version check ok"
}

function patch_diopi_torch() {
  removed_items=""
  added_items=""
  for ((i=0; i<${#torch_4diopi[@]}; i++)); do
      dep_raw=${torch_raws[$i]}
      dep_4diopi=${torch_4diopi[$i]}
      removed_items=${removed_items}" --remove-needed ${dep_raw}"
      added_items=${added_items}" --add-needed  ${dep_4diopi}"
  done
  patchelf ${removed_items} libdiopi_real_impl.so
  patchelf ${added_items} libdiopi_real_impl.so
}

# 1.because dipu libs are loaded by python using dlload. so relative to python main, real_diopi_libs are
# 2-hop dynamic loaded. it cannot see first-hop loaded libs like torch_dipu (unless the lib is loaded 
# using RTLD_GLOBAL, but it's not used when directly loading python lib). so diopi need maunal link 
# torch_dipu.so lib.
# 2.although both the 1st hop dynamic-loaded lib and the 2ed's link to torch_dipu.so, they still share
# the same lib instance in addr space.
function patch_diopi_dipu() {
  patchelf --remove-needed libtorch_dipu.so libdiopi_real_impl.so
  patchelf --add-needed libtorch_dipu.so libdiopi_real_impl.so
}

function remove_unique_symbol() {
  group="torch_$1" # raws or 4diopi
  place="${2:-.}"
  array="$group[@]"
  for name in "${!array}"; do
    echo "PATCH $name"
    python elffile_remove_unique.py -i "${place%%/}/$name"
  done
}

WORK_DIR=$2
cd ${WORK_DIR}
if [[ "$1" == "patch_torch" ]]; then
   gen_versioned_torch
elif [[ "$1" == "patch_diopi" ]]; then
    check_correct_torch $3
    # in dipoi link lib list, torch_dipu.so must be placed behind torch_XX libs. 
    # because both dipu and inner 'DEEPBIND' torch_cpu call Library.<CppFunction>fallback() which is
    # a template function instantiated when parameter types CppFunction is first called
    # (!!! not directly link to the template definition in external torch_cpu.so !!!).
    # if torch_dipu.so is linked in front, <CppFunction>fallback() symbol is bind to the symbol 
    # in torch_dipu.so which use external torch template class that cannot work with inner torch CppFunction.  
    patch_diopi_dipu
    patch_diopi_torch
elif [[ "$1" == "rm_unique_raw" ]]; then
  remove_unique_symbol raws
elif [[ "$1" == "rm_unique_diopi" ]]; then
  remove_unique_symbol 4diopi
fi
