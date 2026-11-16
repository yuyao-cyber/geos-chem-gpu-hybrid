#!/bin/bash
#SBATCH -J cl_setup -p cpu -N 1 --exclusive -o /shared/logs/classic_setup.%j.log
# Classic 4x5 rundir with the SAME configs as the compute1 reference runs (paths rewritten),
# then a GEOS-Chem dry run to list every input file, fetched from s3://geos-chem (same region).
set -euxo pipefail
R=/shared/gc/runs/classic_4x5; S=/shared/gc/run_L_hybrid
mkdir -p $R/Restarts $R/OutputDir
for f in geoschem_config.yml HEMCO_Config.rc HEMCO_Config.rc.gmao_metfields HEMCO_Diagn.rc HISTORY.rc species_database.yml; do cp -f $S/$f $R/; done
cp -f /shared/gc/run_J_gpu/Restarts/GEOSChem.Restart.20140701_0000z.nc4 $R/Restarts/
cd $R
sed -i "s|/path/to/compute1/optimize_gc/MERRA2.20150101.CN.4x5.nc4|/shared/ExtData/GEOS_4x5/MERRA2/2015/01/MERRA2.20150101.CN.4x5.nc4|g; s|/path/to/compute1/optimize_gc/ExtData_met|/shared/ExtData|g; s|/storage1/fs1/rvmartin/Active/GEOS-Chem-shared/ExtData|/shared/ExtData|g" geoschem_config.yml HEMCO_Config.rc HEMCO_Config.rc.gmao_metfields HEMCO_Diagn.rc HISTORY.rc
if grep -n "storage1\|rdcw" geoschem_config.yml HEMCO_Config.rc HEMCO_Config.rc.gmao_metfields HEMCO_Diagn.rc HISTORY.rc; then echo UNREWRITTEN_PATHS; exit 1; fi
grep -n "start_date\|end_date\|root_data_dir\|use_gcclassic_timers" geoschem_config.yml
apptainer exec -B /shared /shared/images/gcdeps-lsf.sif bash -c "cd $R && export PATH=/opt/view/bin:\$PATH LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64:/shared/gc/gpulib:/shared/gc/gpulib/nvrt OMP_NUM_THREADS=8 OMP_STACKSIZE=500m && ulimit -s unlimited && ./gcclassic --dryrun > log.dryrun 2>&1" || true
tail -3 log.dryrun
grep -oE "/shared/ExtData/[^ ,\"'()]+" log.dryrun | grep -v '\$' | sort -u > files_needed.txt
while read f; do [ -e "$f" ] || echo "$f"; done < files_needed.txt > files_missing.txt
echo "needed: $(wc -l < files_needed.txt)  missing: $(wc -l < files_missing.txt)"
: > dl_errors.txt
xargs -a files_missing.txt -P 32 -I{} sh -c 'k=$(echo "{}" | sed "s|^/shared/ExtData/||"); mkdir -p "$(dirname "{}")"; aws s3 cp --quiet "s3://geos-chem/$k" "{}" || echo "NOT_IN_S3 $k" >> dl_errors.txt'
echo "download errors: $(wc -l < dl_errors.txt)"; head -20 dl_errors.txt
du -sh /shared/ExtData
echo CLASSIC_SETUP_DONE
