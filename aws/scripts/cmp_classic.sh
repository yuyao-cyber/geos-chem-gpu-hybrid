#!/bin/bash
# cmp_classic.sh TAG [TEST_CASE...]   -- Classic 1-day accuracy report (runs on head node, /shared/pyenv)
#  A: AWS stock vs compute1 stock reference (bit-identical expected)
#  B: AWS CPU-batch vs AWS stock (bit-identical expected)
#  C: each TEST case (gpu/hyb...) vs stock, judged by the rounding control (fma)
R=/shared/gc/runs/classic_4x5; P=/shared/pyenv/bin/python; F=GEOSChem.Restart.20140702_0000z.nc4
T=$1; shift; REF=/shared/gc/run_J_gpu/case_off/$F
echo "### A: stock on AWS ($T) vs stock on compute1"; $P /shared/gc/compare_restarts.py $R/case_off_$T/$F $REF | grep -E "Summary|DIFF|Only in" | head -12
echo "### B: CPU batch path vs stock (both AWS $T)"; $P /shared/gc/compare_restarts.py $R/case_cpu_$T/$F $R/case_off_$T/$F | grep -E "Summary|DIFF|Only in" | head -12
for c in "$@"; do
  $P /shared/scripts/envelope.py $R/case_off_$T/$F $R/case_fma_$T/$F $R/$c/$F "C: $c vs stock, control = FMA rounding ($T)"
done
