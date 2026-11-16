#!/bin/bash
#SBATCH -J gh_mn -p v100 -N 2 --exclusive -o /shared/logs/gchp_mn.%j.log
# GCHP c24 across NODES (multi-node CPU + GPU). Ranks are started by Slurm (srun --mpi=pmix), each inside
# the gcdeps-ucx container (Open MPI 5.0.5 / PMIx 5). Transport: ob1 + TCP (not UCX-over-TCP, which broke
# MAPL's MPI windows on GCP); shared-memory windows (MPI_LOAD_BALANCE) stay intra-node via osc/sm.
# env: CASES ("off gpu"), PPN ranks per node (6), NX (NY = total/NX, multiple of 6), NDEV GPUs per node (1),
#      FRAC (0.6), STACK (150000), DUR ("00000001 000000"), TAG
set -x
R=/shared/gc/runs/gchp_c24; G=/shared/gc/gpulib; SIF=/shared/images/gcdeps-ucx.sif
NN=$SLURM_JOB_NUM_NODES; PPN=${PPN:-6}; NR=$((NN*PPN)); NX=${NX:-2}; NY=$((NR/NX)); TAG=${TAG:-mn${NN}x${PPN}}
CASES=${CASES:-"off gpu"}
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,tcp OMPI_MCA_osc=sm,rdma OMPI_MCA_btl_tcp_if_exclude=lo,docker0
export OMP_NUM_THREADS=1 NV_ACC_CUDA_STACKSIZE=${STACK:-150000}
ulimit -s unlimited
cd $R; echo "NODES: $SLURM_JOB_NODELIST  ranks=$NR ($PPN/node) NX=$NX NY=$NY TAG=$TAG"
sed -i "s/^JOB_SGMT: .*/JOB_SGMT:     ${DUR:-00000001 000000}/" CAP.rc; grep JOB_SGMT CAP.rc
sed -i "s/^NX: .*/NX: $NX/; s/^NY: .*/NY: $NY/" GCHP.rc; grep -E "^NX:|^NY:" GCHP.rc
srun -N $NN --ntasks-per-node=1 bash -c 'echo "$(hostname): $(nvidia-smi -L | wc -l) GPUs, $(nproc) cpus, AZ $(cat /sys/devices/virtual/dmi/id/board_asset_tag 2>/dev/null)"'
for c in $CASES; do
  D=$R/case_${c}_$TAG; mkdir -p $D
  unset GC_BATCH_CHEM GC_GPU_FRAC GC_GPU_NDEV GC_GPU_FRAC_FIXED
  case $c in off) ;; cpu) export GC_BATCH_CHEM=1 ;; gpu) export GC_BATCH_CHEM=3 GC_GPU_FRAC=${FRAC:-0.6} GC_GPU_NDEV=${NDEV:-1} ;; esac
  echo "20140701 000000" > cap_restart; rm -f Restarts/gcchem_internal_checkpoint OutputDir/*.nc4
  echo "=== CASE $c GC_BATCH_CHEM=${GC_BATCH_CHEM:-unset} NDEV=${GC_GPU_NDEV:-} FRAC=${GC_GPU_FRAC:-} $(date) ==="
  T0=$(date +%s)
  srun --mpi=pmix -N $NN --ntasks-per-node=$PPN --cpu-bind=cores \
    apptainer exec --nv -B /shared $SIF bash -c "export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt:/.singularity.d/libs; ulimit -s unlimited; exec ./gchp" > $D/GC.log 2>&1
  rc=$?; T1=$(date +%s)
  echo "CASE $c rc=$rc elapsed=$((T1-T0)) s  $(date)   cap_restart=$(cat cap_restart)"
  cp -f Restarts/gcchem_internal_checkpoint $D/ 2>/dev/null; cp -f OutputDir/*.nc4 $D/ 2>/dev/null
  grep -a -m3 -E "GPU selftest|GC_GPU_CHEM: nDev|MPI_ERR|invalid window" $D/GC.log
  grep -a "GC_HYB_GCHP: node solve" $D/GC.log | tail -1
  echo "integrate failures: $(grep -a -c "INTEGRATE RETURNED ERROR" $D/GC.log)"
  grep -a "GCHP Date" $D/GC.log | tail -1 | cut -c1-120
done
echo GCHP_MN_DONE
