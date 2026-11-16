#!/bin/bash
#SBATCH -J gpulib -p cpu -N 1 --exclusive -o /shared/logs/gpulib.%j.log
# GPU chemistry library from the model tree's own KPP sources, V100 (cc70) + A100 (cc80) + H100 (cc90).
set -euxo pipefail
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/nvhpc.sif bash "$0"; fi
G=/shared/gc/gpulib; K=/shared/gc/GCClassic-14.7.1-hybrid/src/GEOS-Chem/KPP/fullchem; B=$G/build_aws
export PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin | head -1):$PATH
rm -rf $B; mkdir -p $B $G/nvrt; cd $B
SRC="gckpp_Precision gckpp_Parameters gckpp_JacobianSP gckpp_Function gckpp_Jacobian gckpp_LinearAlgebra gckpp_BatchIntegrator"
for f in $SRC; do cp $K/$f.F90 .; done; cp $G/src/gc_gpu_batch_api.F90 $G/src/gc_gpu_devinfo.c .
md5sum *.F90 *.c | tee SOURCES.md5
FLAGS="-O2 -cpp -r8 -fPIC -tp=haswell -acc -gpu=cc70,cc80,cc90,cuda12.5 -Minfo=accel"
OBJS=""; for f in $SRC gc_gpu_batch_api; do echo "FC $f $(date +%T)"; nvfortran $FLAGS -c $f.F90 -o $f.o > $f.minfo 2>&1 || { tail -20 $f.minfo; exit 1; }; OBJS="$OBJS $f.o"; done
gcc -O2 -fPIC -c gc_gpu_devinfo.c -o gc_gpu_devinfo.o
nvfortran $FLAGS -shared -o libgcchemgpu.so $OBJS gc_gpu_devinfo.o -Wl,-soname,libgcchemgpu.so -Wl,-rpath,$G/nvrt -ldl
cp -f libgcchemgpu.so $G/libgcchemgpu.so
for l in $(ldd $G/libgcchemgpu.so | awk '/=> \/opt\/nvidia/ {print $3}'); do cp -fL $l $G/nvrt/; done
for l in $G/nvrt/*.so*; do for d in $(ldd $l 2>/dev/null | awk '/=> \/opt\/nvidia/ {print $3}'); do [ -f $G/nvrt/$(basename $d) ] || cp -fL $d $G/nvrt/; done; done
nm -D $G/libgcchemgpu.so | grep " T gc_gpu_"; echo GPULIB_OK
