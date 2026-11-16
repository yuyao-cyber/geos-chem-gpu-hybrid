#!/bin/bash
#SBATCH -J gh_res -p cpu -N 1 -o /shared/logs/gchp_month_results.%j.log
# GCHP July-2019 benchmark accuracy from daily checkpoints (read-only on the run folders).
/shared/pyenv/bin/python /shared/scripts/bench/gchp_month_results.py
