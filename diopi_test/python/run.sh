cp /mnt/lustre/wangxing2/DIOPI/impl/torch/fatbin/_fwd_kernel_destindex_copy_kv.fatbin ~/.triton/diopi_triton_kernels.fatbin  
python main.py  --mode=run_test --fname=destindex_copy_kv