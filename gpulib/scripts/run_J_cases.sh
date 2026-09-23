#!/bin/bash
# Workstream C validation + timing: three 1-day 4x5 fullchem runs with the
# SAME binary on the SAME GPU host, back to back:
#   case_off : GC_BATCH_CHEM unset  (stock code path)
#   case_cpu : GC_BATCH_CHEM=1      (CPU batch path, 24 OpenMP threads)
#   case_gpu : GC_BATCH_CHEM=2      (GPU batch path via libgcchemgpu.so)
# Optional 4th case (CASES env var): case_gpu_nosort (=2, GC_BATCH_SORT=0)
# Usage: bash run_J_cases.sh [case list]   default: "off cpu gpu"
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
R=$WS/run_J_gpu
G=$WS/gpulib
CASES=${1:-"gpu off cpu"}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt
export OMP_NUM_THREADS=${NTHREADS:-24}
export OMP_STACKSIZE=500m
export NV_ACC_CUDA_STACKSIZE=160000
ulimit -s unlimited
cd $R
echo "HOST: $(hostname)"; grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L
ldd ./gcclassic | grep -E "gcchem|nvf|acc|gomp|cuda"

for c in $CASES; do
  D=$R/case_$c
  mkdir -p $D
  unset GC_BATCH_CHEM GC_BATCH_SORT
  case $c in
    off)        ;;
    cpu)        export GC_BATCH_CHEM=1 ;;
    gpu)        export GC_BATCH_CHEM=2 ;;
    gpu_nosort) export GC_BATCH_CHEM=2 GC_BATCH_SORT=0 ;;
    gpu_nofma)  export GC_BATCH_CHEM=2; export LD_LIBRARY_PATH=$G/nofma:$LD_LIBRARY_PATH ;;
    cpu_nosort) export GC_BATCH_CHEM=1 GC_BATCH_SORT=0 ;;
  esac
  rm -f Restarts/GEOSChem.Restart.20140702_0000z.nc4 Restarts/HEMCO_restart.201407020000.nc
  rm -f OutputDir/*.nc4
  echo "=== CASE $c  GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset} GC_BATCH_SORT=${GC_BATCH_SORT:-unset}  $(date) ==="
  LD_TRACE_LOADED_OBJECTS=1 ./gcclassic | grep -E "gcchemgpu"   # which solver library the loader resolves (does NOT run the model)
  ( time ./gcclassic > $D/GC.log 2>&1 ) 2> $D/time.log
  rc=$?
  echo "CASE $c rc=$rc  $(date)"
  cp -f Restarts/GEOSChem.Restart.20140702_0000z.nc4 $D/ 2>/dev/null
  cp -f gcclassic_timers.json $D/ 2>/dev/null
  cp -f OutputDir/*.nc4 $D/ 2>/dev/null
  tail -4 $D/GC.log
  grep -m3 -E "GC_BATCH_CHEM|GPU selftest|GC_GPU_CHEM: nDev|GC_GPU_CHEM: selftest" $D/GC.log
  grep "GC_GPU_CHEM: call" $D/GC.log | tail -1
  grep -c "INTEGRATE RETURNED ERROR" $D/GC.log
  cat $D/time.log
done
echo RUN_J_CASES_DONE
