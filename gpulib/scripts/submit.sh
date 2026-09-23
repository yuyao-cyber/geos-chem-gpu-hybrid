#!/bin/bash
# Submission helper (run on the compute1 login node).
#   bash submit.sh libbuild [VARIANT]  -> nvhpc library build (CPU node)
#   bash submit.sh modelbuild [relink] -> gfortran model build (CPU node)
#   bash submit.sh linktest [MODE]     -> gfortran+.so link test, non-399 80GB A100 host
#   bash submit.sh linktest_any [MODE] -> same, any 80GB A100 host (399 allowed)
#   bash submit.sh probe | probe_any   -> CUDA environment diagnostic (deps image)
#   bash submit.sh run  SCRIPT NAME    -> 24-core GPU-host model run (non-399)
#   bash submit.sh run_any SCRIPT NAME -> same, any 80GB A100 host
WS=/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc
G=$WS/gpulib
export LSF_DOCKER_VOLUMES="/storage1/fs1/rvmartin/Active:/storage1/fs1/rvmartin/Active"
DEPS="billzhuge/geos-chem-deps:14.7-lsf"
NVHPC="nvcr.io/nvidia/nvhpc:24.7-devel-cuda12.5-ubuntu22.04"
COMMON="-q general -G compute-rvmartin -g /y.zhuge/bench"
GPUSEL="select[gpuhost && hname!='compute1-exec-399']"
GPUANY="select[gpuhost]"
GPUREQ="num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB"
cd $WS
case "$1" in
  libbuild)
    bsub $COMMON -J gpulib_build -n 4 -R "span[hosts=1] rusage[mem=16GB]" -M 16GB \
      -a "docker($NVHPC)" -o $WS/logs/gpulib_build.%J.log bash $G/build_gpulib.sh $2 ;;
  modelbuild)
    bsub $COMMON -J gpumodel_build -n 16 -R "span[hosts=1] rusage[mem=32GB]" -M 32GB \
      -a "docker($DEPS)" -o $WS/logs/gpumodel_build.%J.log bash $G/build_model_job.sh $2 ;;
  linktest)
    bsub $COMMON -J gpu_linktest -n 2 -R "$GPUSEL span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "$GPUREQ" -a "docker($DEPS)" -o $WS/logs/gpu_linktest.%J.log \
      bash $G/test_link_job.sh $2 ;;
  linktest_any)
    bsub $COMMON -J gpu_linktest -n 2 -R "$GPUANY span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "$GPUREQ" -a "docker($DEPS)" -o $WS/logs/gpu_linktest.%J.log \
      bash $G/test_link_job.sh $2 ;;
  probe)
    bsub $COMMON -J cuda_probe -n 2 -R "$GPUSEL span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "$GPUREQ" -a "docker($DEPS)" -o $WS/logs/cuda_probe.%J.log bash $G/cuda_env_probe.sh ;;
  probe_any)
    bsub $COMMON -J cuda_probe399 -n 2 -R "$GPUANY span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "$GPUREQ" -a "docker($DEPS)" -o $WS/logs/cuda_probe.%J.log bash $G/cuda_env_probe.sh ;;
  probe40)
    bsub $COMMON -J cuda_probe40 -n 2 -R "$GPUSEL span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_40GB" -a "docker($DEPS)" \
      -o $WS/logs/cuda_probe.%J.log bash $G/cuda_env_probe.sh ;;
  probe_nv399)
    bsub $COMMON -J cuda_probe_nv -n 2 -R "$GPUANY span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "$GPUREQ" -a "docker($NVHPC)" -o $WS/logs/cuda_probe_nv.%J.log bash $G/cuda_env_probe.sh ;;
  optview_size)
    bsub $COMMON -J optview_size -n 1 -R "rusage[mem=2GB]" -M 2GB -a "docker($DEPS)" \
      -o $WS/logs/optview_size.%J.log bash -c "du -shL /opt/view; du -sh /opt/spack; ls /opt/view; ls /opt/view/lib | head -30" ;;
  probe_shared)
    bsub $COMMON -J cuda_probe_sh -n 2 -R "$GPUSEL span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "num=1:j_exclusive=no:gmodel=NVIDIAA100_SXM4_80GB" -a "docker($DEPS)" \
      -o $WS/logs/cuda_probe.%J.log bash $G/cuda_env_probe.sh ;;
  run_shared)
    bsub $COMMON -J $3 -n 24 -R "$GPUSEL span[hosts=1] rusage[mem=48GB]" -M 48GB \
      -gpu "num=1:j_exclusive=no:gmodel=NVIDIAA100_SXM4_80GB" -a "docker($DEPS)" \
      -o $WS/logs/$3.%J.log bash $2 ;;
  linktest_shared)
    bsub $COMMON -J gpu_linktest -n 2 -R "$GPUSEL span[hosts=1] rusage[mem=8GB]" -M 8GB \
      -gpu "num=1:j_exclusive=no:gmodel=NVIDIAA100_SXM4_80GB" -a "docker($DEPS)" \
      -o $WS/logs/gpu_linktest.%J.log bash $G/test_link_job.sh $2 ;;
  run)
    bsub $COMMON -J $3 -n 24 -R "$GPUSEL span[hosts=1] rusage[mem=48GB]" -M 48GB \
      -gpu "$GPUREQ" -a "docker($DEPS)" -o $WS/logs/$3.%J.log bash $2 ;;
  run_any)
    bsub $COMMON -J $3 -n 24 -R "$GPUANY span[hosts=1] rusage[mem=48GB]" -M 48GB \
      -gpu "$GPUREQ" -a "docker($DEPS)" -o $WS/logs/$3.%J.log bash $2 ;;
  *) echo "unknown mode: $1"; exit 1 ;;
esac
