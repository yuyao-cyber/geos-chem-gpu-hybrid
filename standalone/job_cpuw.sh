#!/bin/bash
# Phase-1 Exp-2 (warp-cooperative): CPU build + BIT-EXACT gate, mode cpuw, 20,160 cells
set -x
export PATH=/opt/view/bin:$PATH
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
export OMP_NUM_THREADS=24
export OMP_STACKSIZE=512m
bash $D/batch_src/build_batch.sh cpu
cd $D/KPP-Standalone-batch
./kpp_batch_cpu.exe $D/run_H_harvest/lists/random.txt cpuw 20160
echo "JOB_DONE_CPUW"
