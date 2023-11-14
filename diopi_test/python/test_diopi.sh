rm -rf cache
rm -rf gencases

source /mnt/cache/share/platform/env/pt2.0_diopi

srun -p pat_rd --gres=gpu:1 python main.py --mode gen_data --fname $1 --use_db

if [ "$2" == "CAMB" ]; then
    rsync -a --delete cache liangshinan@10.142.4.200:/mnt/lustre/liangshinan/DIOPI/diopi_test/python
    ssh liangshinan@10.142.4.200 "
    source /mnt/cache/share/platform/env/camb_ci_diopi_impl && \
    srun -p camb_mlu370_m8 --gres=mlu:1 bash -c \"cd /mnt/lustre/liangshinan/DIOPI/diopi_test/python && \
    rm -rf gencases && \
    python main.py --mode gen_case --impl_folder ../../impl/camb  --fname $1 --use_db && \
    python main.py --mode run_test --test_cases_path gencases/diopi_case/  --pytest_args='-vs' --use_db\"
    ";
elif [ "$2" == "NV" ]; then
    srun -p pat_rd --gres=gpu:1 bash -c "
        python main.py --mode gen_case --use_db --fname $1 && \
        python main.py --mode run_test --test_cases_path gencases/diopi_case/  --pytest_args='-vs' --use_db";
else
    echo "\$1 与任何条件都不相等"
fi
