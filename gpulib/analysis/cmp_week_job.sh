#!/bin/bash
# Batch job: per-day drift tables for the week runs.
#   bash cmp_week_job.sh hyb      -> four hybrid-vs-stock tables  -> gpulib/drift_week.txt
#   bash cmp_week_job.sh control  -> waits for the FMA control week, then the
#                                    rounding-only table          -> gpulib/drift_control.txt
# Uses the ncio conda env's interpreter EXPLICITLY (conda activate does not
# survive the container's baked PATH).
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
PY=$HOME/miniconda3/envs/ncio/bin/python3
MODE=${1:-hyb}
if [ "$MODE" = "control" ]; then OUT=$WS/gpulib/drift_control.txt; else OUT=$WS/gpulib/drift_week.txt; fi
: > $OUT
echo "HOST $(hostname)  $(date)  MODE=$MODE" >> $OUT
$PY -c "import netCDF4, numpy; print('netCDF4', netCDF4.__version__, 'numpy', numpy.__version__)" >> $OUT 2>&1 || { echo "PYTHON_ENV_BROKEN" >> $OUT; exit 1; }
if [ "$MODE" = "control" ]; then
  for i in $(seq 1 720); do [ -f $WS/run_K_fma_week/case_cpu/GEOSChem.Restart.20140708_0000z.nc4 ] && break; sleep 60; done
  [ -f $WS/run_K_fma_week/case_cpu/GEOSChem.Restart.20140708_0000z.nc4 ] || { echo "CONTROL_NOT_READY" >> $OUT; exit 1; }
  sleep 120   # let the model close its files
  mkdir -p $WS/run_M_week4x5/case_fma
  for f in $WS/run_K_fma_week/case_cpu/GEOSChem.Restart.*.nc4; do ln -sf $f $WS/run_M_week4x5/case_fma/; done
  echo "########## CONTROL run_M_week4x5: off (stock) vs fma (CPU, FMA in KPP lib only) -- rounding-only drift" >> $OUT
  $PY $WS/gpulib/cmp_week.py $WS/run_M_week4x5 off fma >> $OUT 2>&1
else
  echo "### readability check of every restart (open + close)" >> $OUT
  $PY - >> $OUT 2>&1 <<PY
import glob, netCDF4 as nc
bad = 0
for f in sorted(glob.glob("$WS/run_[MN]_*/case_*/GEOSChem.Restart.*.nc4")):
    try: d = nc.Dataset(f); n = len(d.variables); d.close()
    except Exception as e: bad += 1; print("BAD", f, e)
print("unreadable files:", bad)
PY
  for pair in "run_M_week4x5 off hyb1" "run_M_week4x5 off hyb2" "run_N_week2x25 off hyb1" "run_N_week2x25 off hyb2"; do
    set -- $pair
    echo "########## $1: $2 vs $3 (per-day drift)" >> $OUT
    $PY $WS/gpulib/cmp_week.py $WS/$1 $2 $3 >> $OUT 2>&1
  done
fi
echo DRIFT_DONE >> $OUT
