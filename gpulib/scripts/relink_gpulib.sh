#!/bin/bash
# Fast relink of libgcchemgpu.so: recompile ONLY gc_gpu_batch_api.F90 (+ the C
# device-query file) and relink against the KPP objects already in build/.
# The expensive objects (gckpp_Function/Jacobian ~28 min) are reused untouched.
set -e
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
B=$G/build
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
echo "HOST: $(hostname)  START: $(date)"
cd $B
ls -la *.o
cp -f $G/src/gc_gpu_batch_api.F90 $G/src/gc_gpu_devinfo.c $B/
FLAGS="-O2 -cpp -r8 -fPIC -acc -gpu=cc70,cc80,cuda12.5 -Minfo=accel"
nvfortran $FLAGS -c gc_gpu_batch_api.F90 -o gc_gpu_batch_api.o
gcc -O2 -fPIC -c gc_gpu_devinfo.c -o gc_gpu_devinfo.o
OBJS="gckpp_Precision.o gckpp_Parameters.o gckpp_JacobianSP.o gckpp_Function.o \
      gckpp_Jacobian.o gckpp_LinearAlgebra.o gckpp_BatchIntegrator.o \
      gc_gpu_batch_api.o gc_gpu_devinfo.o"
nvfortran $FLAGS -shared -o libgcchemgpu.so $OBJS \
  -Wl,-soname,libgcchemgpu.so -Wl,-rpath,$G/nvrt -ldl
cp -f libgcchemgpu.so $G/libgcchemgpu.so
ls -la $G/libgcchemgpu.so
nm -D $G/libgcchemgpu.so | grep -E " T gc_gpu_"
echo "undefined acc_/omp_ symbols (want: none of acc_get_num_devices etc):"
nm -D -u $G/libgcchemgpu.so | grep -E "acc_|omp_" | head
echo "END: $(date)"
echo RELINK_OK
