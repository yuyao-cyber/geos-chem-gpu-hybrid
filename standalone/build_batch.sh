#!/bin/bash
# Build the batched KPP solver. Usage: build_batch.sh prep|cpu|gpu
# Creates KPP-Standalone-batch/ from KPP-Standalone/ with:
#  - auto-reduce masks (DO_FUN/DO_JVS/DO_SLV) constant-folded to .TRUE.
#  - !$acc routine seq annotations on the four compute kernels
#  - generated interleaved-layout (_I) kernel variants (gckpp_Interleaved.F90)
#  - generated multi-cell (_M) kernel variants  (gckpp_MultiCell{2,4}.F90)
#  - the batched integrator + driver compiled and linked
# NOTE: 'prep' needs python3 (login node).  'cpu'/'gpu' never touch python,
# so they are safe inside the compiler containers.
#
# Environment knobs (compile modes only):
#   BDIR     : build in this directory instead of KPP-Standalone-batch.  The
#              directory must already hold a prepped source tree (rsync it
#              from KPP-Standalone-batch).  Lets a CPU and a GPU build run
#              concurrently without clobbering each other's .o/.mod files.
#   WITH_M4  : 1 (default) also build/link gckpp_MultiCell4 and enable the
#              gpum4/cpum4 driver paths;  0 skips it (much faster build).
set -e
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
SRC=$D/KPP-Standalone
B=${BDIR:-$D/KPP-Standalone-batch}
MODE=${1:-cpu}
WITH_M4=${WITH_M4:-1}
FOPT=${FOPT:--O2}      # optimisation level (the CPU bit-exact gate is
                      # valid at any level: gfortran does not reassociate or
                      # contract FP without -ffast-math/-mfma)

mkdir -p $B
if [ ! -f $B/.patched ]; then
  # copy sources fresh (never touch the validated original tree)
  for f in $SRC/*.F90 $SRC/*.H; do cp -f $f $B/ 2>/dev/null || true; done
  cd $B
  # --- source transformations (masks, scratch array, acc annotations) ---
  python3 $D/batch_src/patch_sources.py
  touch $B/.patched
else
  echo "sources already patched (rm $B/.patched to re-prep)"
fi
cd $B
if [ "$MODE" = "prep" ]; then
  # always refresh our own (non-generated) sources at prep time
  cp -f $D/batch_src/gckpp_BatchIntegrator.F90 $B/
  cp -f $D/batch_src/kpp_batch.F90 $B/
  cp -f $D/batch_src/multicell_integrator.inc $B/
  # generate the interleaved-layout kernels (idempotent; needs python3)
  python3 $D/batch_src/interleave_sources.py
  touch $B/.interleaved
  # generate the multi-cell (per-thread ILP) kernels (idempotent)
  python3 $D/batch_src/multicell_sources.py
  touch $B/.multicell
  exit 0
fi
if [ ! -f $B/.interleaved ] || [ ! -f $B/.multicell ]; then
  echo "ERROR: run 'build_batch.sh prep' on the login node first (needs python3)"
  exit 1
fi
rm -f *.o *.mod   # clean objects when switching compilers

MODS="gckpp_Precision gckpp_Parameters gckpp_Monitor gckpp_JacobianSP \
      gckpp_Global gckpp_Function gckpp_Jacobian gckpp_LinearAlgebra  \
      rateLawUtilFuncs fullchem_RateLawFuncs gckpp_Rates gckpp_Util   \
      gckpp_Initialize gckpp_Integrator kpp_standalone_init           \
      gckpp_Interleaved gckpp_MultiCell2"
M4FLAG=""
if [ "$WITH_M4" = "1" ]; then
  MODS="$MODS gckpp_MultiCell4"
  M4FLAG="-DWITH_M4"
fi
MODS="$MODS gckpp_BatchIntegrator"

if [ "$MODE" = "gpu" ]; then
  FC=nvfortran
  FLAGS="$FOPT -cpp $M4FLAG -acc -gpu=cc70,cc80,cuda12.5,ptxinfo -Minfo=accel"
  EXE=kpp_batch_gpu.exe
else
  FC=gfortran
  FLAGS="$FOPT -cpp $M4FLAG -fopenmp -ffree-line-length-none"
  EXE=kpp_batch_cpu.exe
fi
echo "=== Building $EXE with $FC in $B (WITH_M4=$WITH_M4) ==="

OBJS=""
for f in $MODS; do
  echo "  FC $f.F90   ($(date +%H:%M:%S))"
  $FC $FLAGS -c $f.F90 -o $f.o
  OBJS="$OBJS $f.o"
done
$FC $FLAGS -c kpp_batch.F90 -o kpp_batch.o
$FC $FLAGS kpp_batch.o $OBJS -o $EXE
echo "BUILD_OK $EXE"
