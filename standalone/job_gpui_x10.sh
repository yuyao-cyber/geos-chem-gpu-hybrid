#!/bin/bash
# Phase-1 interleave: SATURATION benchmark (201,600 cells, random_x10.txt)
# mode gpui on A100 (gmodel-constrained submit).  Load takes ~1h; the
# 29,513 c/s gpul plateau (same list, A100) is the comparison baseline.
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
L=$D/run_H_harvest/lists/random_x10.txt
nvidia-smi -L
cd $D/KPP-Standalone-batch
# binary must already exist from the 20k GPU job (no rebuild here)
ls -la kpp_batch_gpu.exe
./kpp_batch_gpu.exe $L gpui
echo "JOB_DONE_GPUI_X10"
