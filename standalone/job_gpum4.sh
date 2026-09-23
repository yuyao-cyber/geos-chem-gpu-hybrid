#!/bin/bash
# Phase-1 Exp-3 (per-thread ILP): NC=4 GPU build.  Builds BOTH MultiCell2 and
# MultiCell4, harvests the ptxinfo per-thread stack frames for both, then runs
#   (1) mode gpum   -- gpul + NC=2   (replicates the NC=2 job on this GPU)
#   (2) mode gpum4  -- gpul + NC=4   (may be infeasible: the CUDA stack is
#       pre-reserved for ~221184 resident thread slots, so a frame of F bytes
#       costs F x 221184 bytes of device memory before any data is allocated)
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
NCELL=${NCELL:-20160}
LIST=${LIST:-$D/run_H_harvest/lists/random.txt}
B=$D/KPP-Standalone-batch
BD=$D/build_gpum4
LOG=$D/logs/build_gpum4.txt
nvidia-smi -L
mkdir -p $BD
cp -f $B/*.F90 $BD/
cp -f $B/*.H   $BD/ 2>/dev/null
cp -f $B/*.inc $BD/
touch $BD/.patched $BD/.interleaved $BD/.multicell
WITH_M4=1 BDIR=$BD bash $D/batch_src/build_batch.sh gpu 2>&1 | tee $LOG
cd $BD

frames () {  # name frame_bytes, one per line
  awk '/Function properties for/{n=$NF} /bytes stack frame/{print n, $1}' $LOG
}
echo "=== ptxinfo: per-thread stack frames (bytes) ==="
frames | sort -u

need () {   # $1 = module tag (multicell2 | multicell4)
  local TOP SUB
  TOP=$(frames | grep "$1" | grep "integrate_cell_m_" | awk '{print $2}' | sort -n | tail -1)
  SUB=$(frames | grep "$1" | grep -E "fun_split_m_|jac_sp_m_|kppdecomp_m_|kppsolve_m_" \
        | awk '{print $2}' | sort -n | tail -1)
  [ -z "$TOP" ] && TOP=0
  [ -z "$SUB" ] && SUB=0
  echo $(( (TOP + SUB) * 11 / 10 + 16384 ))
}
N2=$(need multicell2)
N4=$(need multicell4)
[ "$N2" -lt 160000 ] && N2=160000
[ "$N4" -lt 160000 ] && N4=160000
echo "STACK NC=2 need=$N2  reserve=$(( N2 * 221184 / 1073741824 )) GiB"
echo "STACK NC=4 need=$N4  reserve=$(( N4 * 221184 / 1073741824 )) GiB   (A100-80GB has ~79 GiB usable)"

echo "=== run 1: gpul + NC=2 ==="
NV_ACC_CUDA_STACKSIZE=$N2 ./kpp_batch_gpu.exe $LIST gpum $NCELL || echo "RUN_FAILED_NC2"
echo "=== run 2: gpul + NC=4 ==="
NV_ACC_CUDA_STACKSIZE=$N4 ./kpp_batch_gpu.exe $LIST gpum4 $NCELL || echo "RUN_FAILED_NC4"
echo "JOB_DONE_GPUM4"
