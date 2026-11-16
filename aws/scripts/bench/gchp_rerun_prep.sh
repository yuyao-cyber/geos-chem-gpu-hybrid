#!/bin/bash
#SBATCH -J gh_prep -p cpu -N 1 -t 01:00:00 -o /shared/logs/gchp_rerun_prep.%j.log
# Prepare the GCHP July 2019 rerun (Nov 2026):
#  1. keep the first runs: stock/control/hybrid Restarts checkpoints -> Restarts_run1/, GC.log -> GC_run1.log
#  2. control2 = stock config + initial restart moved one last digit (independent rounding control)
#  3. three 3-hour write tests that write hourly HISTORY files (monthly: 0):
#     writetest_nofix (expected to hang), writetest_fix and writetest_hybrid (expected to pass)
set -ex
B=/shared/bench/gchp; RST=GEOSChem.Restart.20190701_0000z.c24.nc4
for d in stock control hybrid; do
  cd $B/$d
  if [ ! -d Restarts_run1 ]; then mkdir Restarts_run1; mv Restarts/gcchem_internal_checkpoint.* Restarts_run1/; mv GC.log GC_run1.log; fi
  echo "20190701 000000" > cap_restart; ls Restarts | grep -c checkpoint || true
done
newdir() {  # newdir NAME : config copy of stock with its own Restarts/ and OutputDir/
  D=$B/$1; [ -e $D ] && { echo "$D exists, leaving it"; return 1; }
  rsync -a --exclude Restarts --exclude Restarts_run1 --exclude OutputDir --exclude "GC*.log" --exclude allPEs.log $B/stock/ $D/
  mkdir -p $D/Restarts $D/OutputDir; cp -p $B/stock/OutputDir/geoschem_species_metadata.yml $D/OutputDir/ 2>/dev/null || true
  echo "20190701 000000" > $D/cap_restart
}
if newdir control2; then
  cp $B/stock/Restarts/$RST $B/control2/Restarts/$RST
  /shared/pyenv/bin/python /shared/scripts/bench/perturb_restart.py $B/control2/Restarts/$RST
  cmp -s $B/stock/Restarts/$RST $B/control2/Restarts/$RST && { echo "ERROR: restart not perturbed"; exit 1; }
fi
for t in writetest_nofix writetest_fix writetest_hybrid; do
  if newdir $t; then
    ln -s $B/stock/Restarts/$RST $B/$t/Restarts/$RST
    sed -i "s/^\( *[A-Za-z0-9_]*\.monthly: *\)1/\10/" $B/$t/HISTORY.rc
    sed -i "s/^JOB_SGMT: .*/JOB_SGMT:     00000000 030000/" $B/$t/CAP.rc
    echo "$t: monthly=1 left: $(grep -cE "monthly: *1" $B/$t/HISTORY.rc)  $(grep JOB_SGMT $B/$t/CAP.rc)"
  fi
done
ls -la $B; du -sh $B/*/Restarts_run1 2>/dev/null
echo GCHP_RERUN_PREP_DONE
