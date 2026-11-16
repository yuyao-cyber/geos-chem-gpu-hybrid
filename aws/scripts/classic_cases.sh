#!/bin/bash
#SBATCH -J cl_run -N 1 --exclusive -o /shared/logs/classic_run.%j.log
# GEOS-Chem Classic 4x5, 1 day (20140701-02), back-to-back cases with the SAME binary on THIS node.
#   off : stock code path      cpu : GC_BATCH_CHEM=1 (CPU batch, must be bit-identical to stock)
#   fma : rounding-only control exe (KPP with FMA)
#   gpu : GC_BATCH_CHEM=2 (GPU only)      hyb : GC_BATCH_CHEM=3 (CPU+GPU), NDEV GPUs, FRAC start split
# env: CASES, NTHREADS (default: all cores), NDEV (1), FRAC (0.5), STACK (160000), TAG
set -x
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec --nv -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-lsf.sif bash "$0"; fi
R=/shared/gc/runs/classic_4x5; G=/shared/gc/gpulib
CASES=${CASES:-"off cpu hyb"}; TAG=${TAG:-$(hostname -s)}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
# /.singularity.d/libs = host GPU driver libs (libcuda) bound in by apptainer --nv
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt:/.singularity.d/libs
export OMP_NUM_THREADS=${NTHREADS:-$(nproc)} OMP_STACKSIZE=500m NV_ACC_CUDA_STACKSIZE=${STACK:-160000}
ulimit -s unlimited
cd $R; [ -x ./gcclassic ] || { echo NO_EXE; exit 1; }
echo "HOST: $(hostname) TAG=$TAG threads=$OMP_NUM_THREADS"; grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L 2>/dev/null
for c in $CASES; do
  D=$R/case_${c}_$TAG; mkdir -p $D; EXE=./gcclassic
  unset GC_BATCH_CHEM GC_BATCH_SORT GC_GPU_FRAC GC_GPU_FRAC_FIXED GC_GPU_NDEV
  case $c in
    off) ;;
    fma) EXE=$R/fma/gcclassic ;;
    cpu) export GC_BATCH_CHEM=1 ;;
    gpu) export GC_BATCH_CHEM=2 GC_GPU_NDEV=${NDEV:-1} ;;
    hyb) export GC_BATCH_CHEM=3 GC_GPU_FRAC=${FRAC:-0.5} GC_GPU_NDEV=${NDEV:-1} ;;
  esac
  rm -f Restarts/GEOSChem.Restart.20140702_0000z.nc4 Restarts/HEMCO_restart.201407020000.nc OutputDir/*.nc4
  echo "=== CASE $c  EXE=$EXE GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset} NDEV=${GC_GPU_NDEV:-} $(date) ==="
  T0=$(date +%s); $EXE > $D/GC.log 2>&1; rc=$?; T1=$(date +%s)
  echo "CASE $c rc=$rc elapsed=$((T1-T0)) s $(date)"
  cp -f Restarts/GEOSChem.Restart.20140702_0000z.nc4 gcclassic_timers.json OutputDir/*.nc4 $D/ 2>/dev/null
  tail -3 $D/GC.log
  grep -m6 -E "GC_BATCH_CHEM|GPU selftest|GC_GPU_FRAC|GC_HYB_CHEM: using|nDev" $D/GC.log
  grep -E "GC_HYB_CHEM: call|GC_GPU_CHEM: call" $D/GC.log | tail -2
  echo "integrate failures: $(grep -c "INTEGRATE RETURNED ERROR" $D/GC.log)"
done
echo CLASSIC_CASES_DONE
