
function find_triton_kernel() {
    echo "find_triton_kernel"
    rm triton_kernel/ -rvf
    mkdir triton_kernel
    find ~/.triton/cache/ -name '*.ptx' | xargs -I {} sh -c "cp {} ./triton_kernel/ -v; "
    find ~/.triton/cache/ -name '*.cubin' | xargs -I {} cp {} ./triton_kernel/ -v
}

function packect_ptx_and_cubin() {
    echo "packect_ptx_and_cubin"
    cd triton_kernel

    ls *.ptx | awk -F '.' '{print $1}' | xargs -I {} fatbinary -64 --create={}.fatbin "--image3=kind=elf,sm=70,file={}.cubin" "--image3=kind=ptx,sm=70,file={}.ptx"

}

function rename_fatbin() {
    echo "rename_fatbin " $2
    new_file=`cuobjdump --dump-sass  $2 | grep Function | awk '{print  $3 }'`
    echo $new_file
    cp $2 $new_file.fatbin -v
}

case $1 in
    rename_fatbin)
        (
            rename_fatbin $@
        ) \
        || exit -1;;
    find_triton_kernel)
        (
            find_triton_kernel
        ) \
        || exit -1;;
    packect_ptx_and_cubin)
        (
            packect_ptx_and_cubin
        ) \
        || exit -1;;
    *)
        echo -e "default operate:" ;
        find_triton_kernel
        packect_ptx_and_cubin
esac
exit 0
