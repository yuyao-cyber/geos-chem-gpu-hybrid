#!/bin/bash
# Build libgcchemgpu.so with nvfortran (OpenACC) from the MODEL TREE's own
# patched KPP sources + the batch integrator + the bind(C) API.
# Run inside nvcr.io/nvidia/nvhpc:24.7-devel-cuda12.5-ubuntu22.04.
#
# Usage: bash build_gpulib.sh [VARIANT]
#   VARIANT (default "")     : libgcchemgpu.so, default GPU FMA contraction
#   VARIANT "nofma"          : libgcchemgpu_nofma.so (-gpu=nofma)
#   VARIANT "host"           : libgcchemgpu_host.so (-acc=host, CPU fallback)
set -e
VARIANT=${1:-}
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
K=$WS/GCClassic-14.7.1-batch/src/GEOS-Chem/KPP/fullchem
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):/usr/bin:/bin
echo "nvfortran: $(which nvfortran)"; nvfortran --version | head -2
echo "HOST: $(hostname)  START: $(date)"

B=$G/build${VARIANT:+_$VARIANT}
rm -rf $B; mkdir -p $B; cd $B

# --- take the KPP sources FROM THE MODEL TREE (provably the model's mechanism)
KPPFILES="gckpp_Precision gckpp_Parameters gckpp_JacobianSP gckpp_Function \
          gckpp_Jacobian gckpp_LinearAlgebra gckpp_BatchIntegrator"
for f in $KPPFILES; do cp -f $K/$f.F90 $B/; done
cp -f $G/src/gc_gpu_batch_api.F90 $B/
md5sum $B/*.F90 | tee $B/SOURCES.md5

# --- flags: mirror the model's KPP build (-fdefault-real-8 => -r8)
GPUFLAGS="-gpu=cc70,cc80,cuda12.5"
ACC="-acc"
LIB=libgcchemgpu.so
case "$VARIANT" in
  nofma) GPUFLAGS="$GPUFLAGS,nofma"; LIB=libgcchemgpu_nofma.so ;;
  host)  ACC="-acc=host"; GPUFLAGS=""; LIB=libgcchemgpu_host.so ;;
esac
FLAGS="-O2 -cpp -r8 -fPIC $ACC $GPUFLAGS -Minfo=accel"
echo "FLAGS: $FLAGS"

OBJS=""
for f in $KPPFILES gc_gpu_batch_api; do
  echo "  FC $f.F90   ($(date +%H:%M:%S))"
  nvfortran $FLAGS -c $f.F90 -o $f.o
  OBJS="$OBJS $f.o"
done
# NO -mp: the model brings libgomp; do not pull libnvomp in.
nvfortran $FLAGS -shared -o $LIB $OBJS -Wl,-soname,$LIB -Wl,-rpath,$G/nvrt
cp -f $LIB $G/$LIB
echo "=== ldd $LIB ==="
ldd $G/$LIB
# --- copy the nvhpc runtime .so's the library needs so the model can run
#     outside the nvhpc image
mkdir -p $G/nvrt
for l in $(ldd $G/$LIB | awk '/=> \/opt\/nvidia/ {print $3}'); do
  cp -fL $l $G/nvrt/
done
# secondary deps of those runtime libs (e.g. libnvhpcatm, libnvcpumath)
for l in $G/nvrt/*.so*; do
  for d in $(ldd $l 2>/dev/null | awk '/=> \/opt\/nvidia/ {print $3}'); do
    [ -f $G/nvrt/$(basename $d) ] || cp -fL $d $G/nvrt/
  done
done
ls -la $G/nvrt
nm -D $G/$LIB | grep -E " T (gc_gpu_|acc_)" | head
echo "END: $(date)"
echo "BUILD_OK $LIB"
