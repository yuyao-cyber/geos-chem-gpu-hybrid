#!/bin/bash
#SBATCH -J fma -p cpu -N 1 --exclusive -o /shared/logs/build_fma.%j.log
# Rounding-only CONTROL executables: same source, KPP library compiled with FMA contraction (-DKPP_FMA=ON).
# MODEL=classic|gchp  ->  /shared/gc/runs/<rundir>/fma/<exe>
set -euxo pipefail
M=${MODEL:-classic}
if [ $M = classic ]; then SIF=gcdeps-lsf; SRC=/shared/gc/GCClassic-14.7.1-hybrid; R=/shared/gc/runs/classic_4x5; EXE=gcclassic; XO=""
else SIF=gcdeps-ucx; SRC=/shared/gc/GCHP-14.7.1-gpu; R=/shared/gc/runs/gchp_c24; EXE=gchp; XO="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DMPI_LOAD_BALANCE=ON -DOMP=OFF"; fi
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec -B /shared,/var/spool/slurmd --env IN_CONTAINER=1,MODEL=$M /shared/images/$SIF.sif bash "$0"; fi
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64
export UDUNITS2_XML_PATH=/usr/share/xml/udunits ESMF_ROOT=/opt/view CMAKE_PREFIX_PATH=/opt/view
mkdir -p $R/fma $R/build_fma; cd $R/build_fma; rm -rf ./*
cmake $SRC -DRUNDIR=$R/fma $XO -DKPP_FMA=ON -DGC_GPU_LIB=/shared/gc/gpulib/libgcchemgpu.so > cmake.log 2>&1
grep -i "KPP_FMA\|CMake Error" cmake.log || true
make -j32 > make.log 2>&1 || { grep -n -B2 -A8 "Error" make.log | head -40; exit 1; }
f=$(find . -name $EXE -type f -perm -u+x | head -1); cp -f $f $R/fma/$EXE; ls -la $R/fma/$EXE
echo FMA_BUILD_OK $M
