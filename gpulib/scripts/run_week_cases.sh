#!/bin/bash
# Multi-day GEOS-Chem Classic runs, same binary/host, back to back.
#   env RDIR=<rundir>  NTHREADS=<n>   args: case list
#   cases: off | cpu | hyb1 | hyb2 | hyb4
# Waits for the met download marker before starting.
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib; R=${RDIR:?set RDIR}
CASES=${1:-"hyb1 off"}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt
export OMP_NUM_THREADS=${NTHREADS:-16} OMP_STACKSIZE=500m NV_ACC_CUDA_STACKSIZE=160000
ulimit -s unlimited
for i in $(seq 1 720); do [ -f $WS/ExtData_met/dl/DOWNLOAD_OK ] && break; sleep 30; done
[ -f $WS/ExtData_met/dl/DOWNLOAD_OK ] || { echo "MET DOWNLOAD NOT READY"; exit 1; }
cd $R; [ -x ./gcclassic ] || { echo NO_EXE; exit 1; }
echo "HOST: $(hostname)"; grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L
START=$(grep start_date geoschem_config.yml | grep -oE "[0-9]{8}" | head -1)
for c in $CASES; do
  D=$R/case_$c; mkdir -p $D
  unset GC_BATCH_CHEM GC_BATCH_SORT GC_GPU_FRAC GC_GPU_FRAC_FIXED GC_GPU_NDEV
  case $c in
    off) ;;
    cpu)  export GC_BATCH_CHEM=1 ;;
    hyb1) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.5 GC_GPU_NDEV=1 ;;
    hyb2) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.6 GC_GPU_NDEV=2 ;;
    hyb4) export GC_BATCH_CHEM=3 GC_GPU_FRAC=0.7 GC_GPU_NDEV=4 ;;
  esac
  # clear outputs of a previous case (keep the input restart)
  find Restarts -maxdepth 1 -name "GEOSChem.Restart.*.nc4" ! -name "GEOSChem.Restart.${START}_0000z.nc4" -delete
  rm -f Restarts/HEMCO_restart.*.nc OutputDir/*.nc4 HEMCO.log
  echo "=== CASE $c  GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset} NDEV=${GC_GPU_NDEV:-}  $(date) ==="
  ( time ./gcclassic > $D/GC.log 2>&1 ) 2> $D/time.log
  echo "CASE $c rc=$?  $(date)"
  cp -f gcclassic_timers.json HEMCO.log $D/ 2>/dev/null
  find Restarts -maxdepth 1 -name "GEOSChem.Restart.*.nc4" ! -name "GEOSChem.Restart.${START}_0000z.nc4" -exec mv -f {} $D/ \;
  mv -f OutputDir/*.nc4 $D/ 2>/dev/null
  tail -3 $D/GC.log
  grep -m3 -E "GC_BATCH_CHEM|GPU selftest|GC_HYB_CHEM: using" $D/GC.log
  grep -E "GC_HYB_CHEM: call" $D/GC.log | tail -1
  grep -c "INTEGRATE RETURNED ERROR" $D/GC.log
  grep -A14 "G E O S - C H E M   T I M E R S" $D/GC.log | grep -E "GEOS-Chem|chem|Transport|HEMCO"
  cat $D/time.log
done
echo RUN_WEEK_CASES_DONE
