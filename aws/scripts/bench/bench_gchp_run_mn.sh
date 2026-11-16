#!/bin/bash
#SBATCH -J bg_mn -N 2 --exclusive -o /shared/logs/bench_gchp_run_mn.%j.log
# Run one GCHP benchmark case across several nodes. Slurm starts each rank inside the container
# (srun --mpi=pmix); Open MPI uses TCP between nodes and shared memory within a node.
# env: RDIR, MODE = stock | control | hybrid, PPN ranks per node (12), NX (4; NY = total/NX, multiple of 6),
#      NDEV GPUs per node (hybrid), FRAC GPU share (0.75), STACK (150000)
#      COLL Open MPI coll components (default ^han,adapt: han/adapt deadlocked the first HISTORY write, Nov 2026;
#           set COLL=default to run without it), HANG_S watchdog: stop the run if GC.log is silent this long (1800)
# usage: sbatch -p cpubig -N 2 --export=ALL,RDIR=/shared/bench/gchp/control,MODE=control,PPN=12,NX=4 /shared/scripts/bench/bench_gchp_run_mn.sh
set -x
R=${RDIR:?set RDIR}; M=${MODE:?set MODE}; G=/shared/gc/gpulib; SIF=/shared/images/gcdeps-ucx.sif
NN=$SLURM_JOB_NUM_NODES; PPN=${PPN:-12}; NR=$((NN*PPN)); NX=${NX:-4}; NY=$((NR/NX))
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,tcp OMPI_MCA_osc=sm,rdma OMPI_MCA_btl_tcp_if_exclude=lo,docker0
[ "${COLL:-^han,adapt}" != default ] && export OMPI_MCA_coll=${COLL:-^han,adapt}
export OMP_NUM_THREADS=1 NV_ACC_CUDA_STACKSIZE=${STACK:-150000}
ulimit -s unlimited
unset GC_BATCH_CHEM GC_GPU_NDEV GC_GPU_FRAC GC_GPU_FRAC_FIXED
case $M in
  stock)   EXE=/shared/gc/runs/gchp_c24/gchp ;;
  control) EXE=/shared/gc/runs/gchp_c24/fma/gchp ;;
  hybrid)  EXE=/shared/gc/runs/gchp_c24/gchp; export GC_BATCH_CHEM=3 GC_GPU_NDEV=${NDEV:-1} GC_GPU_FRAC=${FRAC:-0.75} ;;
  *) echo "MODE must be stock, control or hybrid"; exit 1 ;;
esac
cd $R || exit 1
sed -i "s/^NX: .*/NX: $NX/; s/^NY: .*/NY: $NY/" GCHP.rc
echo "OMPI_MCA_coll=${OMPI_MCA_coll:-<default>}"
echo "NODES $SLURM_JOB_NODELIST  MODE=$M  ranks=$NR ($PPN/node) NX=$NX NY=$NY  GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset} NDEV=${GC_GPU_NDEV:-}"
echo "cap_restart: $(cat cap_restart)"; grep -E "^JOB_SGMT|^NX:|^NY:|^RECORD_FREQUENCY" CAP.rc GCHP.rc
T0=$(date +%s)
srun --mpi=pmix -N $NN --ntasks-per-node=$PPN --cpu-bind=cores \
  apptainer exec --nv -B /shared $SIF bash -c "export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt:/.singularity.d/libs; ulimit -s unlimited; exec $EXE" > GC.log 2>&1 &
SP=$!; set +x
while kill -0 $SP 2>/dev/null; do sleep 120
  age=$(( $(date +%s) - $(stat -c %Y GC.log) ))
  if [ $age -gt ${HANG_S:-1800} ]; then echo "WATCHDOG: GC.log silent for ${age} s -> stopping the run (hang) $(date)"; tail -3 GC.log
    kill -TERM $SP; sleep 60; kill -KILL $SP 2>/dev/null; break; fi
done
wait $SP; rc=$?; T1=$(date +%s); set -x
echo "BENCH_GCHP_MN $M rc=$rc elapsed=$((T1-T0)) s   cap_restart now: $(cat cap_restart)"
grep -a -m3 -E "GPU selftest|GPU rank|nDev" GC.log
echo "integrate failures: $(grep -a -c "INTEGRATE RETURNED ERROR" GC.log)"
grep -a "GCHP Date" GC.log | tail -1 | cut -c1-120
ls Restarts | tail -5
