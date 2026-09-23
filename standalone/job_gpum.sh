#!/bin/bash
# Phase-1 Exp-3 (per-thread ILP): GPU build + gpul-vs-gpum measurement,
# SAME GPU, SAME job.  NC is selected by MC (2 or 4).
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
MC=${MC:-2}
NCELL=${NCELL:-20160}
LIST=${LIST:-$D/run_H_harvest/lists/random.txt}
B=$D/KPP-Standalone-batch
BD=$D/build_gpum$MC
LOG=$D/logs/build_gpum$MC.txt
nvidia-smi -L
mkdir -p $BD
cp -f $B/*.F90 $BD/
cp -f $B/*.H   $BD/ 2>/dev/null
cp -f $B/*.inc $BD/
touch $BD/.patched $BD/.interleaved $BD/.multicell
if [ "$MC" = "4" ]; then W4=1; MODE=gpum4; else W4=0; MODE=gpum; fi
WITH_M4=$W4 BDIR=$BD bash $D/batch_src/build_batch.sh gpu 2>&1 | tee $LOG
cd $BD

echo "=== ptxinfo: per-thread stack frames (bytes) ==="
awk '/Function properties for/{n=$NF} /bytes stack frame/{print n, $1}' $LOG \
    | sort -u -k2,2n | tail -40

TOP=$(awk '/Function properties for/{n=$NF} /bytes stack frame/{print n, $1}' $LOG \
      | grep -E "integrate_cell_m_" | awk '{print $2}' | sort -n | tail -1)
SUB=$(awk '/Function properties for/{n=$NF} /bytes stack frame/{print n, $1}' $LOG \
      | grep -E "fun_split_m_|jac_sp_m_|kppdecomp_m_|kppsolve_m_" \
      | awk '{print $2}' | sort -n | tail -1)
GPUL=$(awk '/Function properties for/{n=$NF} /bytes stack frame/{print n, $1}' $LOG \
      | grep -E "integrate_cell_l_|integrate_cell_$" | awk '{print $2}' | sort -n | tail -1)
echo "FRAME integrate_cell_m = ${TOP:-?}   worst M kernel = ${SUB:-?}   integrate_cell_l = ${GPUL:-?}"
if [ -z "$TOP" ]; then TOP=0; fi
if [ -z "$SUB" ]; then SUB=0; fi
NEED=$(( (TOP + SUB) * 11 / 10 + 16384 ))
if [ "$NEED" -lt 160000 ]; then NEED=160000; fi
echo "NV_ACC_CUDA_STACKSIZE will be $NEED  (reserved = NEED x ~221184 slots = $(( NEED * 221184 / 1073741824 )) GiB)"

NV_ACC_CUDA_STACKSIZE=$NEED ./kpp_batch_gpu.exe $LIST $MODE $NCELL
echo "JOB_DONE_GPUM$MC"
