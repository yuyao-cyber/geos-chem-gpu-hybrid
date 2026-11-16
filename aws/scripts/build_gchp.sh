#!/bin/bash
#SBATCH -J gchp -p cpu -N 1 --exclusive -o /shared/logs/gchp_build.%j.log
# GCHP GPU tree (LB on, OMP off), linked to the GPU library.
set -euxo pipefail
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-ucx.sif bash "$0"; fi
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64
export UDUNITS2_XML_PATH=/usr/share/xml/udunits ESMF_ROOT=/opt/view CMAKE_PREFIX_PATH=/opt/view
export MPI_ROOT=$(dirname $(dirname $(which mpirun)))
R=/shared/gc/runs/gchp_c24; mkdir -p $R/build; cd $R/build; rm -rf ./*
find /shared/gc/GCHP-14.7.1-gpu -name "*.py" -o -name "*.sh" -o -name "*.pl" | xargs chmod +x || true
cmake /shared/gc/GCHP-14.7.1-gpu -DRUNDIR=$R -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMPI_LOAD_BALANCE=ON -DOMP=OFF -DGC_GPU_LIB=/shared/gc/gpulib/libgcchemgpu.so > cmake.log 2>&1
grep -i "GPU chemistry\|CMake Error\|MPI_LOAD_BALANCE" cmake.log || true
make -j32 install > make.log 2>&1 && echo GCHP_BUILD_OK || { grep -n -B2 -A8 "Error" make.log | head -40; exit 1; }
ls -la $R/gchp
