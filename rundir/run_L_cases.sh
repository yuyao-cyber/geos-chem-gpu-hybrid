#!/bin/bash
# Hybrid-tree validation + timing, same binary, same GPU host, back to back.
#   cpu : GC_BATCH_CHEM=1  (regression: must stay bit-identical to stock)
#   hyb : GC_BATCH_CHEM=3  (CPU+GPU concurrent)
#   gpu : GC_BATCH_CHEM=2  (GPU only, for reference)
#   off : stock
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
R=${RDIR:-$WS/run_L_hybrid}; G=$WS/gpulib
CASES=${1:-"cpu hyb"}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt
export OMP_NUM_THREADS=${NTHREADS:-24} OMP_STACKSIZE=500m NV_ACC_CUDA_STACKSIZE=160000
ulimit -s unlimited
cd $R
[ -x ./gcclassic ] || { echo NO_EXE; exit 1; }
echo "HOST: $(hostname)"; grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L
LD_TRACE_LOADED_OBJECTS=1 ./gcclassic | grep -E "gcchemgpu|gomp"
for c in $CASES; do
  D=$R/case_$c; mkdir -p $D
  unset GC_BATCH_CHEM GC_BATCH_SORT GC_GPU_FRAC GC_GPU_FRAC_FIXED GC_GPU_NDEV
  case $c in
    off) ;;
    cpu) export GC_BATCH_CHEM=1 ;;
    gpu) export GC_BATCH_CHEM=2 ;;
    hyb) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.4 ;;
    hyb1) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.5 GC_GPU_NDEV=1 ;;
    hyb2) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.6 GC_GPU_NDEV=2 ;;
    hyb2fix) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.95 GC_GPU_FRAC_FIXED=1 GC_GPU_NDEV=2 ;;
    hyb2mid) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.85 GC_GPU_FRAC_FIXED=1 GC_GPU_NDEV=2 ;;
    hyb4) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.7 GC_GPU_NDEV=4 ;;
    hyb_nosort) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.4 GC_BATCH_SORT=0 ;;
  esac
  rm -f Restarts/GEOSChem.Restart.20140702_0000z.nc4 Restarts/HEMCO_restart.201407020000.nc OutputDir/*.nc4
  echo "=== CASE $c  GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset}  $(date) ==="
  ( time ./gcclassic > $D/GC.log 2>&1 ) 2> $D/time.log
  echo "CASE $c rc=$?  $(date)"
  cp -f Restarts/GEOSChem.Restart.20140702_0000z.nc4 gcclassic_timers.json OutputDir/*.nc4 $D/ 2>/dev/null
  tail -3 $D/GC.log
  grep -m6 -E "GC_BATCH_CHEM|GPU selftest|GC_GPU_FRAC|GC_HYB_CHEM: using|nDev" $D/GC.log
  grep -E "GC_HYB_CHEM: call|GC_GPU_CHEM: call" $D/GC.log | tail -2
  grep -c "INTEGRATE RETURNED ERROR" $D/GC.log
  cat $D/time.log
done
echo RUN_L_CASES_DONE
