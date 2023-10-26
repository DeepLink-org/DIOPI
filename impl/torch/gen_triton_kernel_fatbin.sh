function rename_ptx_kernel() {
    echo $ptx_path

}

function find_triton_kernel() {
    echo "find_triton_kernel"
    rm triton_kernel/ -rvf
    mkdir triton_kernel
    find ~/.triton/cache/ -name '*.ptx' | xargs -I {} sh -c "cp {} ./triton_kernel/ ; "
    find ~/.triton/cache/ -name '*.cubin' | xargs -I {} cp {} ./triton_kernel/ -v
}

function packect_ptx_and_cubin() {
    echo "packect_ptx_and_cubin"
    cd triton_kernel

    ls *.ptx | awk -F '.' '{print $1}' | xargs -I {} fatbinary -64 --create={}.fatbin "--image3=kind=elf,sm=70,file={}.cubin" "--image3=kind=ptx,sm=70,file={}.ptx"

}

case $1 in
    find_triton_kernel)
        (
            find_triton_kernel
        ) \
        || exit -1;;
    packect_ptx_and_cubin)
        (
            find_triton_kernel
            packect_ptx_and_cubin
        ) \
        || exit -1;;
    *)
        echo -e "[ERROR] Incorrect option:" $1;
esac
exit 0
