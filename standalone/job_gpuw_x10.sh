#!/bin/bash
# Phase-1 Exp-2 (warp-cooperative): SATURATION benchmark, 201,600 cells.
# Run ONLY if the 20,160-cell gpuw sweep is competitive with gpul.
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
L=$D/run_H_harvest/lists/random_x10.txt
nvidia-smi -L
cd $D/KPP-Standalone-batch
ls -la kpp_batch_gpu.exe
NV_ACC_CUDA_STACKSIZE=160000 ./kpp_batch_gpu.exe $L gpuw
echo "JOB_DONE_GPUW_X10"
