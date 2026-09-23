#!/bin/bash
# Relink libgcchemgpu_nofma.so from the nofma KPP objects (build_nofma/) and
# ADD gc_gpu_devinfo.o (gc_gpu_info), which build_gpulib.sh predates.
set -e
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib; B=$G/build_nofma
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
cd $B; ls *.o
cp -f $G/src/gc_gpu_devinfo.c $B/ && gcc -O2 -fPIC -c gc_gpu_devinfo.c -o gc_gpu_devinfo.o
FLAGS="-O2 -cpp -r8 -fPIC -acc -gpu=cc70,cc80,cuda12.5,nofma"
OBJS="gckpp_Precision.o gckpp_Parameters.o gckpp_JacobianSP.o gckpp_Function.o gckpp_Jacobian.o gckpp_LinearAlgebra.o gckpp_BatchIntegrator.o gc_gpu_batch_api.o gc_gpu_devinfo.o"
nvfortran $FLAGS -shared -o libgcchemgpu_nofma.so $OBJS -Wl,-soname,libgcchemgpu.so -Wl,-rpath,$G/nvrt -ldl
cp -f libgcchemgpu_nofma.so $G/libgcchemgpu_nofma.so
nm -D $G/libgcchemgpu_nofma.so | grep -E " T gc_gpu_"
echo RELINK_NOFMA_OK
