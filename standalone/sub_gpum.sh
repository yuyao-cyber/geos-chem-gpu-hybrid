#!/bin/bash
# Submit the Exp-3 GPU job.  Usage: bash sub_gpum.sh [MC] [NCELL] [LIST]
D=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
export MC=${1:-2}
export NCELL=${2:-20160}
export LIST=${3:-$D/run_H_harvest/lists/random.txt}
export LSF_DOCKER_VOLUMES="/storage1/fs1/rvmartin/Active:/storage1/fs1/rvmartin/Active"
bsub -G compute-rvmartin -g /y.zhuge/bench -q general -n 4 \
  -R "select[gpuhost] rusage[mem=96GB] span[hosts=1]" \
  -R "select[hname!='compute1-exec-399']" \
  -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" \
  -a "docker(nvcr.io/nvidia/nvhpc:24.7-devel-cuda12.5-ubuntu22.04)" \
  -o $D/logs/gpum${MC}_${NCELL}.%J.log \
  bash $D/batch_src/job_gpum.sh
