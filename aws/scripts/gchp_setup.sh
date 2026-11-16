#!/bin/bash
#SBATCH -J gh_setup -p cpu -N 1 --exclusive -o /shared/logs/gchp_setup.%j.log
# GCHP c24 rundir with the SAME configs as the compute1 run_GH_gpu runs (paths rewritten),
# inputs resolved from ExtData.rc for 2014-07-01..02 and fetched from s3://geos-chem.
set -euxo pipefail
R=/shared/gc/runs/gchp_c24; S=/shared/gc/run_GH_gpu; X=/shared/ExtData
mkdir -p $R/Restarts $R/OutputDir
for f in CAP.rc ESMF.rc ExtData.rc GCHP.rc HEMCO_Config.rc HEMCO_Diagn.rc HISTORY.rc input.nml logging.yml \
         geoschem_config.yml species_database.yml kpp_standalone_interface.yml setCommonRunSettings.sh; do cp -f $S/$f $R/; done
cp -f $S/Restarts/GEOSChem.Restart.fullchem.20140701_0000z.c24.nc4 $R/Restarts/
cd $R
sed -i "s|/storage1/fs1/rvmartin/Active/GEOS-Chem-shared/ExtData|$X|g" geoschem_config.yml HEMCO_Config.rc
if grep -n "storage1\|rdcw" *.rc *.yml; then echo UNREWRITTEN_PATHS; exit 1; fi
ln -nsf $X/GEOS_0.25x0.3125/GEOS_FP MetDir; ln -nsf $X/CHEM_INPUTS ChemDir; ln -nsf $X/HEMCO HcoDir
ln -nsf Restarts/GEOSChem.Restart.fullchem.20140701_0000z.c24.nc4 gchp_restart.nc4
echo "20140701 000000" > cap_restart
grep -n "JOB_SGMT\|NX:\|NY:\|IM_WORLD" CAP.rc GCHP.rc
# inputs
cd /shared/gc/runs/gchp_data
python3 /shared/scripts/gchp_resolve_extdata.py $R/ExtData.rc 20140701 20140702 gchp 2> resolve.err
: > dl_errors.txt
xargs -a gchp_missing.txt -P 32 -I{} sh -c 'mkdir -p "$(dirname "'$X'/{}")"; aws s3 cp --quiet "s3://geos-chem/{}" "'$X'/{}" || echo "FAIL {}" >> dl_errors.txt'
# directories read directly by GEOS-Chem (not through ExtData)
aws s3 sync --quiet s3://geos-chem/CHEM_INPUTS/CLOUD_J/v2025-01/ $X/CHEM_INPUTS/CLOUD_J/v2025-01/
aws s3 sync --quiet s3://geos-chem/CHEM_INPUTS/Aerosol_Optics/v2025-03/ $X/CHEM_INPUTS/Aerosol_Optics/v2025-03/
echo "download errors: $(wc -l < dl_errors.txt)"; head dl_errors.txt
n=0; while read k s; do [ -s "$X/$k" ] || { n=$((n+1)); echo "ABSENT $k"; }; done < gchp_needed.txt; echo "absent after download: $n"
du -sh $X
echo GCHP_SETUP_DONE
