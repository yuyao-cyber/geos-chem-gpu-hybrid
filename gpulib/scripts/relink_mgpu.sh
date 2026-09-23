#!/bin/bash
# Fast relink: compile the NEW gc_gpu_batch_api.F90 (adds
# gc_gpu_integrate_batch_dev) against the existing KPP objects in build/ and
# produce libgcchemgpu_mgpu.so (soname libgcchemgpu.so, drop-in via
# LD_LIBRARY_PATH=$G/mgpu).  Run inside the nvhpc image.
set -e
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib; B=$G/build
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
echo "HOST: $(hostname)  START: $(date)"
cd $B
cp -f $G/src/gc_gpu_batch_api.F90 $B/gc_gpu_batch_api_mgpu.F90
cp -f $G/src/gc_gpu_devinfo.c $B/
FLAGS="-O2 -cpp -r8 -fPIC -tp=haswell -acc -gpu=cc70,cc80,cuda12.5 -Minfo=accel"
grep -n "acc_device_nvidia" /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/include/openacc.h | head -3
for l in libacchost libaccdevice libaccdevaux; do echo "$l: $(nm -D /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/lib/$l.so 2>/dev/null | grep -c \" T acc_set_device_num\")"; done
nvfortran $FLAGS -c gc_gpu_batch_api_mgpu.F90 -o gc_gpu_batch_api_mgpu.o
gcc -O2 -fPIC -c gc_gpu_devinfo.c -o gc_gpu_devinfo.o
OBJS="gckpp_Precision.o gckpp_Parameters.o gckpp_JacobianSP.o gckpp_Function.o \
      gckpp_Jacobian.o gckpp_LinearAlgebra.o gckpp_BatchIntegrator.o \
      gc_gpu_batch_api_mgpu.o gc_gpu_devinfo.o"
nvfortran $FLAGS -shared -o libgcchemgpu_mgpu.so $OBJS \
  -Wl,-soname,libgcchemgpu.so -Wl,-rpath,$G/nvrt -ldl
cp -f libgcchemgpu_mgpu.so $G/libgcchemgpu_mgpu.so
mkdir -p $G/mgpu && ln -sfn ../libgcchemgpu_mgpu.so $G/mgpu/libgcchemgpu.so
# nvhpc runtime libs the library needs (e.g. libnvcpumath-avx2.so from -tp=haswell)
for l in $(ldd $G/libgcchemgpu_mgpu.so | awk '/=> \/opt\/nvidia/ {print $3}'); do cp -fL $l $G/nvrt/; done
for l in $G/nvrt/*.so*; do for d in $(ldd $l 2>/dev/null | awk '/=> \/opt\/nvidia/ {print $3}'); do [ -f $G/nvrt/$(basename $d) ] || cp -fL $d $G/nvrt/; done; done
echo "ldd (want no 'not found'):"; LD_LIBRARY_PATH=$G/nvrt ldd $G/libgcchemgpu_mgpu.so | grep -E "not found|nvcpumath" 
nm -D $G/libgcchemgpu_mgpu.so | grep -E " T gc_gpu_"
echo "undefined acc_/omp_ symbols (want none):"; nm -D -u $G/libgcchemgpu_mgpu.so | grep -E "acc_|omp_" | head
echo "END: $(date)"; echo RELINK_MGPU_OK
