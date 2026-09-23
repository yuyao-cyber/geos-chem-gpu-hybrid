#!/bin/bash
# cmp_cases.sh <restartA> <restartB> <label>  -- per-variable + robust diff stats
# Runs on the login node in the ncio conda env. Output: gpulib/cmp_<label>.txt and gpulib/diff_<label>.txt
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
K=$WS/GCClassic-14.7.1-batch/src/GEOS-Chem/KPP/fullchem/gckpp_Monitor.F90
A=$1; B=$2; LBL=$3
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ncio
python3 $G/compare_gpu_restarts.py $A $B $K > $G/cmp_$LBL.txt 2>&1; echo "exit=$?" >> $G/cmp_$LBL.txt
python3 $G/analyze_gpu_diff.py $A $B > $G/diff_$LBL.txt 2>&1
echo "=== $LBL: summary ==="; grep -E "^Summary|^class|^KPP-var|^KPP-fix|^nonKPP|^Chem_|^other" $G/cmp_$LBL.txt | head -8
echo "=== $LBL: robust stats ==="; sed -n 1,30p $G/diff_$LBL.txt
