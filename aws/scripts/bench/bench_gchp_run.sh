#!/bin/bash
#SBATCH -J bg_run -N 1 --exclusive -o /shared/logs/bench_gchp_run.%j.log
# Run one GCHP benchmark case in its own run directory (duration from CAP.rc / setCommonRunSettings.sh).
# env: RDIR, MODE = stock | control | hybrid, NRANKS (24), NX (4; NY = NRANKS/NX must be a multiple of 6),
#      NDEV GPUs (4), FRAC GPU share of cells (0.75), STACK (150000)
# usage: sbatch -p v100 --export=ALL,RDIR=/shared/bench/gchp/hybrid,MODE=hybrid,NRANKS=24,NX=4,NDEV=4,FRAC=0.75 /shared/scripts/bench/bench_gchp_run.sh
set -x
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec --nv -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-ucx.sif bash "$0"; fi
R=${RDIR:?set RDIR}; G=/shared/gc/gpulib; M=${MODE:?set MODE}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt:/.singularity.d/libs
export OMP_NUM_THREADS=1 NV_ACC_CUDA_STACKSIZE=${STACK:-150000}
ulimit -s unlimited
NR=${NRANKS:-24}; NX=${NX:-4}; NY=$((NR/NX))
unset GC_BATCH_CHEM GC_GPU_NDEV GC_GPU_FRAC GC_GPU_FRAC_FIXED
case $M in
  stock)   EXE=/shared/gc/runs/gchp_c24/gchp ;;
  control) EXE=/shared/gc/runs/gchp_c24/fma/gchp ;;
  hybrid)  EXE=/shared/gc/runs/gchp_c24/gchp; export GC_BATCH_CHEM=3 GC_GPU_NDEV=${NDEV:-4} GC_GPU_FRAC=${FRAC:-0.75} ;;
  *) echo "MODE must be stock, control or hybrid"; exit 1 ;;
esac
cd $R || exit 1
sed -i "s/^NX: .*/NX: $NX/; s/^NY: .*/NY: $NY/" GCHP.rc
echo "HOST $(hostname)  MODE=$M  ranks=$NR NX=$NX NY=$NY  GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset} NDEV=${GC_GPU_NDEV:-} FRAC=${GC_GPU_FRAC:-}"
grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L 2>/dev/null
echo "cap_restart: $(cat cap_restart)"; grep -E "^JOB_SGMT|^NX:|^NY:|^RECORD_FREQUENCY" CAP.rc GCHP.rc
T0=$(date +%s); mpirun -np $NR $EXE > GC.log 2>&1; rc=$?; T1=$(date +%s)
echo "BENCH_GCHP $M rc=$rc elapsed=$((T1-T0)) s   cap_restart now: $(cat cap_restart)"
grep -a -m3 -E "GPU selftest|GPU rank|nDev" GC.log
grep -a "GC_HYB_GCHP: node solve" GC.log | tail -1
echo "integrate failures: $(grep -a -c "INTEGRATE RETURNED ERROR" GC.log)"
grep -a "GCHP Date" GC.log | tail -1 | cut -c1-120
ls Restarts | tail -5
