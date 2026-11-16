#!/bin/bash
#SBATCH -J gh_run -N 1 --exclusive -o /shared/logs/gchp_run.%j.log
# GCHP c24 back-to-back cases in /shared/gc/runs/gchp_c24 (same layout as compute1 run_gchp_cases.sh).
# env: CASES ("off gpu"), NRANKS (6), NX (1; NY=NRANKS/NX, must be a multiple of 6), NDEV (1), FRAC (0.5), STACK (160000), DUR ("00000001 000000"), TAG
set -x
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec --nv -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-ucx.sif bash "$0"; fi
R=${RDIR:-/shared/gc/runs/gchp_c24}; G=/shared/gc/gpulib
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
# /.singularity.d/libs = host GPU driver libs (libcuda) bound in by apptainer --nv
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt:/.singularity.d/libs
export OMP_NUM_THREADS=1 NV_ACC_CUDA_STACKSIZE=${STACK:-160000}
ulimit -s unlimited
NR=${NRANKS:-6}; CASES=${CASES:-"off gpu"}; TAG=${TAG:-$(hostname -s)}
cd $R; echo "HOST: $(hostname)  TAG=$TAG"
sed -i "s/^JOB_SGMT: .*/JOB_SGMT:     ${DUR:-00000001 000000}/" CAP.rc; grep JOB_SGMT CAP.rc
NX=${NX:-1}; NY=$((NR/NX)); sed -i "s/^NX: .*/NX: $NX/; s/^NY: .*/NY: $NY/" GCHP.rc; grep -E "^NX:|^NY:" GCHP.rc
grep -m1 "model name" /proc/cpuinfo; nproc; nvidia-smi -L 2>/dev/null
[ -x ./gchp ] || { echo NO_EXE; exit 1; }
for c in $CASES; do
  D=$R/case_${c}_$TAG; mkdir -p $D; EXE=./gchp
  unset GC_BATCH_CHEM GC_GPU_FRAC GC_GPU_NDEV GC_GPU_FRAC_FIXED
  case $c in off) ;; fma) EXE=$R/fma/gchp ;; cpu) export GC_BATCH_CHEM=1 ;; gpu) export GC_BATCH_CHEM=3 GC_GPU_FRAC=${FRAC:-0.5} GC_GPU_NDEV=${NDEV:-1} ;; esac
  echo "20140701 000000" > cap_restart; rm -f Restarts/gcchem_internal_checkpoint* OutputDir/*.nc4
  echo "=== CASE $c  EXE=$EXE GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset}  $(date) ==="
  T0=$(date +%s); mpirun -np $NR $EXE > $D/GC.log 2>&1; rc=$?; T1=$(date +%s)
  echo "CASE $c rc=$rc elapsed=$((T1-T0)) s  $(date)   cap_restart=$(cat cap_restart)"
  cp -f Restarts/gcchem_internal_checkpoint* $D/ 2>/dev/null; cp -f OutputDir/*.nc4 $D/ 2>/dev/null
  grep -m4 -E "GC_BATCH_CHEM|GPU selftest|GC_HYB|GC_GPU_CHEM: nDev" $D/GC.log
  grep -E "GC_HYB_GCHP" $D/GC.log | tail -3
  grep -c "INTEGRATE RETURNED ERROR" $D/GC.log
  grep -E "GCHP Date|Model Throughput" $D/GC.log | tail -2
  tail -3 $D/GC.log
done
echo RUN_GCHP_CASES_DONE
