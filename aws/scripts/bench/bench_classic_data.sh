#!/bin/bash
#SBATCH -J bc_data -p cpu -N 1 --exclusive -o /shared/logs/bench_classic_data.%j.log
# Download every input a GEOS-Chem Classic run directory needs (from a dry run) from s3://geos-chem.
# usage: sbatch --export=ALL,RDIR=/shared/bench/classic/stock /shared/scripts/bench/bench_classic_data.sh
set -x
R=${RDIR:?set RDIR}; X=/shared/ExtData; G=/shared/gc/gpulib
cd $R || exit 1
apptainer exec -B /shared /shared/images/gcdeps-lsf.sif bash -c "cd $R && export PATH=/opt/view/bin:\$PATH LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:$G:$G/nvrt OMP_NUM_THREADS=8 OMP_STACKSIZE=500m && ulimit -s unlimited && /shared/gc/runs/classic_4x5/gcclassic --dryrun > log.dryrun 2>&1"
tail -3 log.dryrun
grep -oE "$X/[^ ,\"'()]+" log.dryrun | sed 's|//*|/|g' | grep -v '\$' | grep -v '/$' | sort -u > files_needed.txt
while read f; do [ -e "$f" ] || echo "$f"; done < files_needed.txt > files_missing.txt
echo "needed: $(wc -l < files_needed.txt)  missing: $(wc -l < files_missing.txt)"
: > dl_errors.txt
xargs -a files_missing.txt -P 32 -I{} sh -c 'k=$(echo "{}" | sed "s|^/shared/ExtData/||"); mkdir -p "$(dirname "{}")"; aws s3 cp --quiet "s3://geos-chem/$k" "{}" || echo "NOT_IN_S3 $k" >> dl_errors.txt'
n=0; while read f; do [ -e "$f" ] || { n=$((n+1)); echo "ABSENT $f"; }; done < files_needed.txt
echo "download errors: $(wc -l < dl_errors.txt)   absent after download: $n"
du -sh $X
echo BENCH_CLASSIC_DATA_DONE
