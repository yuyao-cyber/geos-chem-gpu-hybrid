#!/bin/bash
# Diagnose cuInit failure in the deps image on a GPU host.
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt
echo "HOST: $(hostname)"
nvidia-smi -L
ls -la /dev/nvidia* 2>&1
env | grep -i -E "nvidia|cuda|acc" | sort
cat /proc/driver/nvidia/version 2>&1
ls -la /usr/lib/x86_64-linux-gnu/libcuda* /usr/lib/x86_64-linux-gnu/libnvidia-ptxjit* /usr/lib/x86_64-linux-gnu/libnvidia-nvvm* 2>&1
mkdir -p $G/build_test && cd $G/build_test
echo "=== plain C driver API (gcc, no nvhpc runtime) ==="
gcc -O0 -o cuinit_test $G/src/cuinit_test.c -ldl && ./cuinit_test; echo "rc=$?"
echo "=== same with CUDA_VISIBLE_DEVICES unset ==="
env -u CUDA_VISIBLE_DEVICES ./cuinit_test; echo "rc=$?"
echo "=== nvhpc-built standalone (kpp_batch_gpu.exe) under the deps image ==="
ls -la $WS/KPP-Standalone-batch/kpp_batch_gpu.exe
export NV_ACC_CUDA_STACKSIZE=160000
head -3 $WS/run_H_harvest/*.txt 2>/dev/null | head -5
LIST=$WS/run_H_harvest/lists/sorted_x10.txt
echo "LIST=$LIST"
if [ -n "$LIST" ]; then
  cd $WS/KPP-Standalone-batch && timeout 300 ./kpp_batch_gpu.exe $LIST gpul 64 2>&1 | tail -8
fi
echo "=== test_link.exe with NV_ACC_NOTIFY=31 / ACC_DEVICE_TYPE ==="
cd $G/build_test
NV_ACC_NOTIFY=31 ./test_link.exe 2>&1 | tail -5
ACC_DEVICE_TYPE=host ./test_link.exe 2>&1 | tail -3
echo "=== ltrace-free: strace of open/dlopen calls around cuInit ==="
which strace && strace -f -e trace=openat -o /dev/stdout ./test_link.exe 2>&1 | grep -E "nvidia|cuda" | head -20
echo PROBE_DONE
