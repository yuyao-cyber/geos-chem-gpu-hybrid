#!/bin/bash
# Phase-1 scoping: (1) launch-shape sweep, (2) profiler runs to identify
# the kernel's true limiter (occupancy/registers vs bandwidth vs latency).
set -x
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/profilers/*/target-linux-x64 2>/dev/null | head -1):$PATH
export NV_ACC_CUDA_STACKSIZE=160000
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
B=$D/KPP-Standalone-batch
L=$D/run_H_harvest/lists/random.txt   # 20,160 cells: fast loads for many runs
cd $B
nvidia-smi -L | head -1
FLAGS="-O2 -cpp -acc -gpu=cc70,cc80,cuda12.5"

echo "=== baseline (default vector length) ==="
./kpp_batch_gpu.exe $L gpul 20160

for VL in 64 256; do
  echo "=== vector_length($VL) ==="
  sed "s/!\$acc parallel loop gang vector\$/!\$acc parallel loop gang vector vector_length($VL)/" kpp_batch.F90 > kpp_batch_vl.F90
  nvfortran $FLAGS -c kpp_batch_vl.F90 -o kpp_batch_vl.o 2>&1 | tail -2
  nvfortran $FLAGS kpp_batch_vl.o gckpp_BatchIntegrator.o gckpp_Integrator.o \
    kpp_standalone_init.o gckpp_Initialize.o gckpp_Util.o gckpp_Rates.o \
    fullchem_RateLawFuncs.o rateLawUtilFuncs.o gckpp_LinearAlgebra.o \
    gckpp_Jacobian.o gckpp_Function.o gckpp_Global.o gckpp_JacobianSP.o \
    gckpp_Monitor.o gckpp_Parameters.o gckpp_Precision.o -o kpp_batch_vl$VL.exe
  ./kpp_batch_vl$VL.exe $L gpul 20160
done

echo "=== register/occupancy info from compiler ==="
nvfortran $FLAGS -gpu=ptxinfo -c gckpp_BatchIntegrator.F90 -o /tmp/bi.o 2>&1 | grep -iE "registers|spill|Function" | head -20

echo "=== Nsight Compute attempt (may be blocked by cluster policy) ==="
which ncu && timeout 1200 ncu --launch-count 1 --kernel-name-base demangled --set basic ./kpp_batch_gpu.exe $L gpul 2000 2>&1 | tail -40 || echo "NCU_UNAVAILABLE_OR_BLOCKED"
