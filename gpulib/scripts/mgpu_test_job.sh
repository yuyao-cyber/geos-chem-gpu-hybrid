#!/bin/bash
# Multi-GPU functional test of gc_gpu_integrate_batch_dev (deps image, GPU host, >=2 GPUs).
# Samples nvidia-smi utilization during the run so host fallback cannot masquerade as success.
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$G/mgpu:/opt/view/lib:/opt/view/lib64:$G:$G/nvrt
export NV_ACC_CUDA_STACKSIZE=160000
echo "HOST: $(hostname)"; grep -m1 "model name" /proc/cpuinfo; nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
mkdir -p $G/build_test && cd $G/build_test
gfortran -O2 -fopenmp -o mgpu_test.exe $G/src/mgpu_test.F90 $G/libgcchemgpu_mgpu.so \
  -Wl,-rpath,$G/mgpu -Wl,-rpath,$G/nvrt -Wl,--export-dynamic-symbol=acc_register_library || exit 1
LD_TRACE_LOADED_OBJECTS=1 ./mgpu_test.exe | grep gcchemgpu
rm -f gpu_util.csv
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > gpu_util.csv &
SMI=$!
NV_ACC_NOTIFY=1 ./mgpu_test.exe 100000 2>&1 | grep -v "^launch CUDA kernel" | head -60; echo "rc=${PIPESTATUS[0]}"
# how many kernel launches went to each device (NV_ACC_NOTIFY=1 prints one line per launch)
NV_ACC_NOTIFY=1 ./mgpu_test.exe 20000 2>&1 | grep "^launch CUDA kernel" | sed -E 's/.*device=([0-9]+).*/device=\1/' | sort | uniq -c
kill $SMI 2>/dev/null
echo "=== max GPU utilization / memory per device during the run (nvidia-smi samples) ==="
awk -F', ' '{u[$1]=($2>u[$1])?$2:u[$1]; m[$1]=($3>m[$1])?$3:m[$1]} END {for (d in u) print "gpu", d, "max_util%=" u[d], "max_mem_MiB=" m[d]}' gpu_util.csv
echo MGPU_TEST_DONE
