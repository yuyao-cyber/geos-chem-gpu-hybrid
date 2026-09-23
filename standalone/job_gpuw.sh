#!/bin/bash
# Phase-1 Exp-2 (warp-cooperative): GPU build + gpul-vs-gpuw lane sweep, SAME GPU
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
L=$D/run_H_harvest/lists/random.txt
nvidia-smi -L
bash $D/batch_src/build_batch.sh gpu 2>&1 | tee $D/logs/build_gpuw.txt
cd $D/KPP-Standalone-batch
echo "=== -Minfo vector-loop evidence inside Integrate_Cell_W / KppDecomp_W ==="
grep -n -A2 -B2 "integrate_cell_w\|kppdecomp_w\|addghinvdiag_w" $D/logs/build_gpuw.txt | head -80
echo "=== gpuw sweep (NV_ACC_CUDA_STACKSIZE=160000) ==="
NV_ACC_CUDA_STACKSIZE=160000 ./kpp_batch_gpu.exe $L gpuw 20160
echo "JOB_DONE_GPUW"
