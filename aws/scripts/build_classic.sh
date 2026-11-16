#!/bin/bash
#SBATCH -J classic -p cpu -N 1 --exclusive -o /shared/logs/classic_build.%j.log
# GEOS-Chem Classic hybrid tree (stock path bit-identical), linked to the GPU library.
set -euxo pipefail
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-lsf.sif bash "$0"; fi
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64
R=/shared/gc/runs/classic_4x5; mkdir -p $R/build $R/Restarts $R/OutputDir; cd $R/build; rm -rf ./*
cmake /shared/gc/GCClassic-14.7.1-hybrid -DRUNDIR=$R -DGC_GPU_LIB=/shared/gc/gpulib/libgcchemgpu.so > cmake.log 2>&1
grep -i "GPU chemistry\|CMake Error" cmake.log || true
make -j32 install > make.log 2>&1 && echo CLASSIC_BUILD_OK || { grep -n -B2 -A8 "Error" make.log | head -40; exit 1; }
ls -la $R/gcclassic
