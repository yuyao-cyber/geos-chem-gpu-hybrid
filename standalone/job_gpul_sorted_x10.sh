#!/bin/bash
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
export NV_ACC_CUDA_STACKSIZE=160000
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
nvidia-smi -L
cd $D/KPP-Standalone-batch
ls -la kpp_batch_gpu.exe
echo "=== gpul RANDOM x10 (in-job baseline) ==="
./kpp_batch_gpu.exe $D/run_H_harvest/lists/random_x10.txt gpul
echo "=== gpul SORTED x10 ==="
./kpp_batch_gpu.exe $D/run_H_harvest/lists/sorted_x10.txt gpul
echo "JOB_DONE_GPUL_SORTED_X10"
