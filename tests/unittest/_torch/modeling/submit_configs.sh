#!/bin/bash

# a config consists of TP, CP, EP, dense (boolean), ctx len (0 for sweep), batch
# configs="8,1,2,0,0 16,1,4,0,0 32,1,8,0,0 1,8,2,0,0 1,16,4,0,0 1,32,8,0,0 2,4,2,0,0 2,8,4,0,0 2,16,8,0,0"
# configs+=" 8,1,2,1,0 16,1,4,1,0 32,1,8,1,0 1,8,2,1,0 1,16,4,1,0 1,32,8,1,0 2,4,2,1,0 2,8,4,1,0 2,16,8,1,0"
configs="1,1,1,1,1048576,8 2,1,1,1,1048576,8 4,1,1,1,1048576,8 8,1,1,1,1048576,8 16,1,1,1,1048576,8 32,1,1,1,1048576,8 64,1,1,1,1048576,8"
configs+=" 8,1,1,1,32768,8 8,1,1,1,65536,8 8,1,1,1,131072,8 8,1,1,1,262144,8 8,1,1,1,524288,8 8,1,1,1,1048576,8 8,1,1,1,2097152,8 8,1,1,1,4194304,8 8,1,1,1,8388608,8 8,1,1,1,16777216,8"
configs+=" 8,1,1,1,1048576,8 8,2,1,1,1048576,8 8,4,1,1,1048576,8 8,8,1,1,1048576,8"

for config in $configs; do
    IFS=","
    set -- $config
    tp=$1
    cp=$2
    ep=$3
    dense=$4
    ctx_len=$5
    batch=$6
    echo "TP $tp, CP $cp, EP $ep, dense $dense, ctx len $ctx_len, batch $batch, profiling"
    bash test_helix_deepseek_sbatch.sh $tp $cp $ep $dense $ctx_len $batch "nvfp4" 1
done
