# SUPA Support

BR Fullstask must be installed in order to support SUPA device.

1. uninstall exsting fullstack if need. e.g, upgrade fullstack.
    ```
    cd full-stack
    sudo ./clean.sh -p all
    ```
2. select and download BR fullstack based on running OS type.
3. install fullstack, e.g.

    ```
    tar -xzf full-stack_ubuntu-18.04_master_1112_20230918.tar.gz
    cd full-stack
    sudo -E ./install.sh -p base
    source /etc/profile.d/biren.sh
    ```

4. install SUPA diopi_impl, e.g.
    ```
    cd full-stack
    dpkg -i biren-diopi-0.1.0-amd64.deb
    ```
5. compile DIOPI / DIPU
    ```
    export DIOPI_SUPA_DIR=/usr/local/supa
    export DIOPI_IMPL_LIB=${DIOPI_SUPA_DIR}/lib

    cd third_party/DIOPI/impl
    export DIOPI_BUILD_TESTRT=ON
    sh scripts/build_impl.sh clean
    sh scripts/build_impl.sh supa

    sh template_build_sh builddl supa supa
    sh template_build_sh builddp
    ```
6. run DIOPI / DIPU
    ```
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DIOPI_SUPA_DIR}/lib
    export DIPU_DEVICE_MEMCACHING_ALGORITHM="RAW"
    ```
