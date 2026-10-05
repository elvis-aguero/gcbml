#!/bin/bash
# usage: bench_schemes.sh : s/iter for vmap, shard (4 host devices), threads at n_pad 120 and 368, 3 repeats
for n in 120 368; do
  for s in vmap shard threads; do
    if [ $s = shard ]; then export GCBML_HOST_DEVICES=4; else unset GCBML_HOST_DEVICES; fi
    echo -n "$s "; GCBML_CHAIN_SCHEME=$s uv run python scripts/bench_chains.py $n 3 2>&1 | tail -1
  done
done
