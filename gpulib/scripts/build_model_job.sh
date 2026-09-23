#!/bin/bash
# Build GC-Classic (gfortran, deps image) in run_J_gpu/build with the GPU
# library linked in (-DGC_GPU_LIB).  Waits for libgcchemgpu.so to appear.
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
R=$WS/run_J_gpu
G=$WS/gpulib
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64
echo "HOST: $(hostname)  START: $(date)"
for i in $(seq 1 120); do
  [ -f $G/libgcchemgpu.so ] && grep -q BUILD_OK $G/build/../logs_marker 2>/dev/null && break
  [ -f $G/libgcchemgpu.so ] && [ -f $G/nvrt/libnvf.so ] && break
  sleep 30
done
[ -f $G/libgcchemgpu.so ] || { echo NO_LIB; exit 1; }
ls -la $G/libgcchemgpu.so $G/nvrt | head
cd $R/build
[ "$1" = "relink" ] || rm -rf $R/build/*
cmake ../CodeDir -DRUNDIR=.. -DGC_GPU_LIB=$G/libgcchemgpu.so > cmake.log 2>&1
grep -n -i "GPU chemistry\|GC_GPU\|CMake Error" cmake.log | head
make -j16 install > make.log 2>&1 && echo BUILD_OK
grep -n -i -B2 -A6 "error" make.log | head -80
tail -5 make.log
ls -la $R/gcclassic
ldd $R/gcclassic | grep -E "gcchem|nv|gomp|cuda|gfortran"
echo "END: $(date)"
echo MODEL_BUILD_DONE
