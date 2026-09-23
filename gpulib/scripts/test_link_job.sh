#!/bin/bash
# Link test: gfortran (deps image) main + nvfortran libgcchemgpu.so on a GPU host.
# Run inside billzhuge/geos-chem-deps:14.7-lsf with a GPU allocated by LSF.
#   MODE=direct : link libgcchemgpu.so directly; the executable defines a
#                 no-op acc_register_library (see test_link.F90)   (default)
#   MODE=shim   : build + link the RTLD_DEEPBIND dlopen shim (record only:
#                 DEEPBIND breaks cuInit, and cannot fix the weak-symbol hook)
set -x
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
LIB=${LIB:-libgcchemgpu.so}
MODE=${1:-direct}
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt
export NV_ACC_CUDA_STACKSIZE=160000
echo "HOST: $(hostname)"; nvidia-smi -L
mkdir -p $G/build_test; cd $G/build_test
rm -f *.o *.mod test_link.exe omp_then_gpu.exe
if [ "$MODE" = "shim" ]; then
  gcc -O2 -fPIC -shared -o $G/libgcchemgpu_shim.so $G/src/gc_gpu_shim.c \
      -DGC_GPU_LIB_DEFAULT="\"$G/$LIB\"" -ldl || exit 1
  LINKLIB=$G/libgcchemgpu_shim.so
else
  LINKLIB=$G/$LIB
fi
gfortran -O2 -fopenmp -o test_link.exe $G/src/test_link.F90 $LINKLIB \
  -Wl,-rpath,$G -Wl,-rpath,$G/nvrt -Wl,--export-dynamic-symbol=acc_register_library
ldd ./test_link.exe | grep -E "gcchem|nv|cuda|gomp|gfortran"
nm -D test_link.exe | grep acc_register_library
./test_link.exe
echo "rc=$?"
# a threaded caller, as in the model (OpenMP regions around the GPU call)
cat > omp_then_gpu.F90 <<'EOF'
PROGRAM omp_then_gpu
  USE ISO_C_BINDING
  IMPLICIT NONE
  INTERFACE
     SUBROUTINE gc_gpu_selftest( rc, nsteps, tTotal ) BIND(C, name='gc_gpu_selftest')
       IMPORT :: c_int, c_double
       INTEGER(c_int), INTENT(OUT) :: rc, nsteps
       REAL(c_double), INTENT(OUT) :: tTotal
     END SUBROUTINE gc_gpu_selftest
  END INTERFACE
  INTEGER :: i, n, rc, nst, tid, tmax
  REAL(c_double) :: t
  INTEGER, EXTERNAL :: omp_get_thread_num, omp_get_num_threads
  n = 0; tmax = 0
  !$OMP PARALLEL DO REDUCTION(+:n) REDUCTION(max:tmax) PRIVATE(tid)
  DO i = 1, 1000
     n = n + 1
     tid = omp_get_thread_num(); tmax = MAX(tmax, tid)
  ENDDO
  !$OMP END PARALLEL DO
  CALL gc_gpu_selftest( rc, nst, t )
  !$OMP PARALLEL DO REDUCTION(+:n) REDUCTION(max:tmax) PRIVATE(tid)
  DO i = 1, 1000
     n = n + 1
     tid = omp_get_thread_num(); tmax = MAX(tmax, tid)
  ENDDO
  !$OMP END PARALLEL DO
  WRITE(*,'(a,i0,a,i0,a,i0,a,i0)') 'omp_then_gpu: n=', n, ' rc=', rc, ' nsteps=', nst, ' max_tid=', tmax
  IF ( rc == 1 .and. n == 2000 .and. tmax == 7 ) WRITE(*,'(a)') 'OMP+GPU coexistence OK'
END PROGRAM omp_then_gpu
SUBROUTINE acc_register_library( reg, unreg, lookup ) BIND(C, name='acc_register_library')
  USE ISO_C_BINDING, ONLY : C_PTR
  TYPE(C_PTR), VALUE :: reg, unreg, lookup
END SUBROUTINE acc_register_library
EOF
gfortran -O2 -fopenmp -o omp_then_gpu.exe omp_then_gpu.F90 $LINKLIB -Wl,-rpath,$G -Wl,-rpath,$G/nvrt -Wl,--export-dynamic-symbol=acc_register_library
nm -D omp_then_gpu.exe | grep acc_register_library
OMP_NUM_THREADS=8 ./omp_then_gpu.exe
echo "rc=$?"
echo "=== LD_DEBUG: bindings FROM nvhpc/gpu libs TO libgomp/libgfortran/exe (want: none or benign) ==="
OMP_NUM_THREADS=8 LD_DEBUG=bindings ./omp_then_gpu.exe 2>&1 | grep "binding file" \
  | grep -E "file [^ ]*(nvrt|gcchemgpu)" | grep -E "to [^ ]*(libgomp|libgfortran|omp_then_gpu)" \
  | sed -E 's/.*binding file ([^ ]*) .* to ([^ ]*) .*symbol `([^'"'"']*).*/\1 -> \2 : \3/' | sort | uniq -c | head -20
echo "=== LD_DEBUG: bindings FROM exe/libgomp/libgfortran TO nvhpc libs (want: only gc_gpu_*) ==="
OMP_NUM_THREADS=8 LD_DEBUG=bindings ./omp_then_gpu.exe 2>&1 | grep "binding file" \
  | grep -E "file [^ ]*(omp_then_gpu|libgomp|libgfortran)" | grep -E "to [^ ]*(nvrt|gcchemgpu)" \
  | sed -E 's/.*binding file ([^ ]*) .* to ([^ ]*) .*symbol `([^'"'"']*).*/\1 -> \2 : \3/' | sort | uniq -c | head -20
echo TEST_LINK_DONE
