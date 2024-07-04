

#!/usr/bin/env bash
set -ex 

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


LIBS_DIR=$2
cd ${LIBS_DIR}
if [[ "$1" == "patch_torch" ]]; then
   gen_versioned_torch
elif [[ "$1" == "patch_diopi" ]]; then
    # dipu lib must linked after torch lib
    patch_diopi_dipu
    patch_diopi_torch
    
fi