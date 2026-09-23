#!/bin/bash
# Phase-1 interleave: GPU build + gpul-vs-gpui comparison on the SAME GPU
# (20,160-cell primary benchmark; gpul run in the same job = same-model baseline)
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
L=$D/run_H_harvest/lists/random.txt
nvidia-smi -L
bash $D/batch_src/build_batch.sh gpu
cd $D/KPP-Standalone-batch

echo "=== gpul baseline (NV_ACC_CUDA_STACKSIZE=160000) ==="
NV_ACC_CUDA_STACKSIZE=160000 ./kpp_batch_gpu.exe $L gpul 20160

echo "=== gpui, default stack ==="
./kpp_batch_gpu.exe $L gpui 20160

echo "=== gpui, NV_ACC_CUDA_STACKSIZE=160000 ==="
NV_ACC_CUDA_STACKSIZE=160000 ./kpp_batch_gpu.exe $L gpui 20160

echo "JOB_DONE_GPUI"
