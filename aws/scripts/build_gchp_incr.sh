#!/bin/bash
#SBATCH -J gchp_incr -p cpu -N 1 --exclusive -o /shared/logs/gchp_incr.%j.log
# Incremental rebuild of the existing GCHP GPU build tree after a source edit; installs build/bin/gchp
# into the rundir (previous executable kept as gchp.prev).
set -euxo pipefail
if [ -z "${IN_CONTAINER:-}" ]; then exec apptainer exec -B /shared,/var/spool/slurmd --env IN_CONTAINER=1 /shared/images/gcdeps-ucx.sif bash "$0"; fi
export PATH=/opt/view/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/opt/view/lib:/opt/view/lib64
R=/shared/gc/runs/gchp_c24; cd $R/build
make -j32 > make_incr.log 2>&1 || { grep -n -B2 -A8 "Error" make_incr.log | head -40; exit 1; }
cp -a $R/gchp $R/gchp.prev; cp -f $R/build/bin/gchp $R/gchp; ls -la $R/gchp $R/gchp.prev
echo GCHP_INCR_OK
