#!/bin/bash
# cmp_gchp.sh TAG [TEST_CASE_DIR...]  -- GCHP c24 1-day accuracy report (head node, /shared/pyenv)
#  A: AWS stock path vs compute1 stock checkpoint (bit-identical expected)
#  B: AWS CPU-batch path vs AWS stock (bit-identical expected, if case_cpu exists)
#  C: each TEST case vs stock, judged by the rounding control (case_fma)
R=/shared/gc/runs/gchp_c24; P=/shared/pyenv/bin/python; F=gcchem_internal_checkpoint
T=$1; shift; REF=/shared/gc/run_GH_gpu/day1_v100/ckpt_off.nc4
echo "### A: stock on AWS ($T) vs stock on compute1"; $P /shared/gc/c1tools/gpulib/cmp_gchp_ckpt.py $R/case_off_$T/$F $REF | head -3
[ -d $R/case_cpu_$T ] && { echo "### B: CPU batch vs stock ($T)"; $P /shared/gc/c1tools/gpulib/cmp_gchp_ckpt.py $R/case_cpu_$T/$F $R/case_off_$T/$F | head -3; }
for c in "$@"; do
  $P /shared/scripts/envelope.py $R/case_off_$T/$F $R/case_fma_$T/$F $R/$c/$F "C: $c vs stock, control = FMA rounding ($T)"
done
