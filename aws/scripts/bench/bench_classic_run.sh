#!/bin/bash
#SBATCH -J bc_run -N 1 --exclusive -o /shared/logs/bench_classic_run.%j.log
# Run one GEOS-Chem Classic benchmark case in its own run directory (start/end from geoschem_config.yml).
# env: RDIR, MODE = stock | control | hybrid, NTHREADS (24), NDEV GPUs (1), STACK (150000)
# usage: sbatch -p v100 --export=ALL,RDIR=/shared/bench/classic/hybrid,MODE=hybrid,NTHREADS=24,NDEV=2 /shared/scripts/bench/bench_classic_run.sh
set -x
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec --nv -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-lsf.sif bash "$0"; fi
R=${RDIR:?set RDIR}; G=/shared/gc/gpulib; M=${MODE:?set MODE}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt:/.singularity.d/libs
export OMP_NUM_THREADS=${NTHREADS:-24} OMP_STACKSIZE=500m NV_ACC_CUDA_STACKSIZE=${STACK:-150000}
ulimit -s unlimited
unset GC_BATCH_CHEM GC_GPU_NDEV GC_GPU_FRAC GC_GPU_FRAC_FIXED
case $M in
  stock)   EXE=/shared/gc/runs/classic_4x5/gcclassic ;;
  control) EXE=/shared/gc/runs/classic_4x5/fma/gcclassic ;;
  hybrid)  EXE=/shared/gc/runs/classic_4x5/gcclassic; export GC_BATCH_CHEM=3 GC_GPU_NDEV=${NDEV:-1} GC_GPU_FRAC=${FRAC:-0.5} ;;
  *) echo "MODE must be stock, control or hybrid"; exit 1 ;;
esac
cd $R || exit 1
echo "HOST $(hostname)  MODE=$M  threads=$OMP_NUM_THREADS  GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset}  NDEV=${GC_GPU_NDEV:-}"
grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L 2>/dev/null
grep -E "start_date|end_date" geoschem_config.yml
T0=$(date +%s); $EXE > GC.log 2>&1; rc=$?; T1=$(date +%s)
echo "BENCH_CLASSIC $M rc=$rc elapsed=$((T1-T0)) s"
tail -3 GC.log
grep -m3 -E "GPU selftest|GC_HYB_CHEM: using|nDev" GC.log
grep -E "GC_HYB_CHEM: call" GC.log | tail -1
echo "integrate failures: $(grep -c "INTEGRATE RETURNED ERROR" GC.log)"
ls Restarts | tail -5
