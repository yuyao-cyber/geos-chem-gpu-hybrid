#!/bin/bash
# Build the HYBRID tree (GC_BATCH_CHEM=3 support) with the GPU library linked.
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
R=$WS/run_L_hybrid; G=$WS/gpulib
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64
echo "HOST: $(hostname)  START: $(date)"
cd $R/build && rm -rf $R/build/*
cmake ../CodeDir -DRUNDIR=.. -DGC_GPU_LIB=$G/libgcchemgpu.so > cmake.log 2>&1
grep -n -i "GPU chemistry\|CMake Error" cmake.log | head
make -j16 install > make.log 2>&1 && echo BUILD_OK
grep -n -i -B2 -A8 "error" make.log | grep -v "Werror\|-Wno-error\|errorCount\|ErrMsg\|GC_Error\|error_mod\|ErrorMsg\|errors.\|error_stop" | head -80
ls -la $R/gcclassic && ldd $R/gcclassic | grep -E "gcchem|gomp"
echo "END: $(date)"; echo HYBRID_BUILD_DONE
