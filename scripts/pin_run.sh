#!/bin/bash
# usage: pin_run.sh NCPU cmd... : run cmd restricted to the first NCPU cpus of the allocation
n=$1; shift
cpus=$(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{ if (NF==2) for(i=$1;i<=$2;i++) print i; else print $1 }' | head -$n | paste -sd,)
exec taskset -c $cpus "$@"
