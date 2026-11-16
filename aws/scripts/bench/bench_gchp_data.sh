#!/bin/bash
#SBATCH -J bg_data -p cpu -N 1 --exclusive -o /shared/logs/bench_gchp_data.%j.log
# Download every input a GCHP run directory needs for START..END (from its ExtData.rc + direct reads).
# usage: sbatch --export=ALL,RDIR=/shared/bench/gchp/stock,START=20190701,END=20190731 /shared/scripts/bench/bench_gchp_data.sh
set -x
R=${RDIR:?set RDIR}; S=${START:-20190701}; E=${END:-20190731}; X=/shared/ExtData
W=/shared/bench/gchp_data; mkdir -p $W; cd $W
python3 /shared/scripts/gchp_resolve_extdata.py $R/ExtData.rc $S $E bench 2> resolve.err
tail -4 resolve.err
: > dl_errors.txt
xargs -a bench_missing.txt -P 32 -I{} sh -c 'mkdir -p "$(dirname "'$X'/{}")"; aws s3 cp --quiet "s3://geos-chem/{}" "'$X'/{}" || echo "FAIL {}" >> dl_errors.txt'
# HEMCO volcano tables (read directly, one file per day)
d=$(date -d "$S - 1 day" +%Y%m%d); last=$(date -d "$E + 2 day" +%Y%m%d)
while [ "$d" -le "$last" ]; do k=HEMCO/VOLCANO/v2024-04/${d:0:4}/${d:4:2}/so2_volcanic_emissions_Carns.$d.rc
  [ -s $X/$k ] || { mkdir -p $(dirname $X/$k); aws s3 cp --quiet s3://geos-chem/$k $X/$k || echo "FAIL $k" >> dl_errors.txt; }
  d=$(date -d "$d + 1 day" +%Y%m%d); done
# restart file(s) the run directory links to
for f in $R/Restarts/*.nc4 $R/gchp_restart.nc4; do [ -L "$f" ] || continue; t=$(readlink -f "$f" 2>/dev/null || readlink "$f")
  case "$t" in $X/*) [ -s "$t" ] || { mkdir -p $(dirname $t); aws s3 cp --quiet "s3://geos-chem/${t#$X/}" "$t" || echo "FAIL ${t#$X/}" >> dl_errors.txt; } ;; esac; done
n=0; while read k s; do [ -s "$X/$k" ] || { n=$((n+1)); echo "ABSENT $k"; }; done < bench_needed.txt
echo "download errors: $(wc -l < dl_errors.txt)   absent after download: $n"; head dl_errors.txt
du -sh $X
echo BENCH_GCHP_DATA_DONE
