#!/bin/bash
# Phase-1 Exp-3 (per-thread ILP): CPU build + BIT-EXACT gate, mode cpum,
# 20,160 cells, BOTH cell->thread pairings, NC=2 and NC=4.
set -x
export PATH=/opt/view/bin:$PATH
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
B=$D/KPP-Standalone-batch
BD=$D/build_cpum
export OMP_NUM_THREADS=24
export OMP_STACKSIZE=1024m
mkdir -p $BD
cp -f $B/*.F90 $BD/
cp -f $B/*.H   $BD/ 2>/dev/null
cp -f $B/*.inc $BD/
touch $BD/.patched $BD/.interleaved $BD/.multicell
WITH_M4=1 BDIR=$BD bash $D/batch_src/build_batch.sh cpu
cd $BD
ulimit -s unlimited
./kpp_batch_cpu.exe $D/run_H_harvest/lists/random.txt cpum 20160
echo "JOB_DONE_CPUM"
