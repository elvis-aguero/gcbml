#!/bin/bash
# usage: par_probe.sh NPROC PIN(0/1) FLAGS... : run NPROC concurrent bench_step.py 368, optionally one core each
n=$1; pin=$2; shift 2
export XLA_FLAGS="$*"
cpus=($(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{ if (NF==2) for(i=$1;i<=$2;i++) print i; else print $1 }'))
for i in $(seq 0 $((n-1))); do
  if [ "$pin" = 1 ]; then pre="taskset -c ${cpus[$i]}"; else pre=""; fi
  ($pre .venv/bin/python scripts/bench_step.py 368 2>&1 | tail -1 | sed "s/^/p$i /") &
done
wait
