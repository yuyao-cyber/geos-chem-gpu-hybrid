#!/bin/bash
#SBATCH -J both -N 1 --exclusive -o /shared/logs/both.%j.log
# Classic AND GCHP on the SAME GPU node (identical hardware/config for the model comparison).
# Whatever GPU queue has capacity runs it; TAG records the instance type.
# env: GPUS_USED (1), CL_CASES, GH6_CASES, GH12_CASES
set -x
IT=$(curl -s -m 5 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)
[ -z "$IT" ] && { T=$(curl -s -m 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60"); IT=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/instance-type); }
AZ=$(T=$(curl -s -m 5 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60"); curl -s -m 5 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/placement/availability-zone)
TAGB=$(echo ${IT:-unknown} | tr -d '.')g${GPUS_USED:-1}
echo "INSTANCE $IT  AZ $AZ  TAG $TAGB  $(date)"; nvidia-smi -L; nproc; free -g | head -2
export NDEV=${GPUS_USED:-1} FRAC=${FRAC:-0.5}
CASES="${CL_CASES:-off fma cpu hyb gpu}" NTHREADS=${NTHREADS:-16} TAG=$TAGB bash /shared/scripts/classic_cases.sh
CASES="${GH6_CASES:-off fma cpu gpu}" NRANKS=6 NX=1 TAG=${TAGB}r6 bash /shared/scripts/gchp_cases.sh
[ -n "${GH12_CASES-off fma gpu}" ] && CASES="${GH12_CASES:-off fma gpu}" NRANKS=12 NX=2 TAG=${TAGB}r12 bash /shared/scripts/gchp_cases.sh
echo "BOTH_DONE $TAGB $(date)"
