#!/bin/bash

# a config consists of CP, TP, EP, batch_size
# ORIGINAL FROM MATTHIAS:configs="8,1,2 16,1,4 32,1,8 1,8,2 1,16,4 1,32,8 2,4,2 2,8,4 2,16,8"
# # ck-4_ctp8ep8, ck-16_cep64, ck-8_cep64, ck-4_cep64, ck-2_ctp8cep4tep2, ck-1_ctp32tp2
# configs="32,2,2,16 32,2,2,32 32,2,2,16 32,2,2,32 32,2,2,16 32,2,2,32"
# ck-1_ep64, ck-32_ctp32tp2, ck-16_ctp32tp2, ck-8_ctp32tp2, ck-4_ctp32tp2, ck-2_ctp8cep4tep2, ck-1_ctp32tp2, ck-1_tp64
#configs="1,64,64,64,true 32,2,2,32,false 32,2,2,16,false 32,2,2,8,false 32,2,2,4,false 32,2,2,2,false 32,2,8,2,false 32,2,2,1,false"
configs="1,64,8,128,true 1,64,8,256,true 1,64,8,64,true"
ctx_len_start=$((2 ** 20))
ctx_len_end=$((2 ** 20))

for config in $configs; do
    IFS=","
    set -- $config
    cp=$1
    tp=$2
    ep=$3
    batch_size=$4
    attention_dp=$5
    echo "CP $cp, TP $tp, EP $ep, Batch size $batch_size"
    ctx_len=$ctx_len_start
    while [ $ctx_len -le $ctx_len_end ]; do
        if [[ $ctx_len -ge $((2 ** 24)) && $cp -eq 1 ]]; then
            echo "skipping ctx_len $ctx_len because it is too long for cp 1"
        else
            echo "Batch size $batch_size"
            sed -i "s/isl=[[:digit:]]\+/isl=$ctx_len/" submit_mjoux.sh
            sed -i "s/gen_tp_size=[[:digit:]]\+/gen_tp_size=$tp/" submit_mjoux.sh
            sed -i "s/gen_cp_size=[[:digit:]]\+/gen_cp_size=$cp/" submit_mjoux.sh
            sed -i "s/gen_ep_size=[[:digit:]]\+/gen_ep_size=$ep/" submit_mjoux.sh
            sed -i "s/batch=[[:digit:]]\+/batch=$batch_size/" submit_mjoux.sh
            sed -i "s/enable_attention_dp=[[:alpha:]]\+/enable_attention_dp=$attention_dp/" submit_mjoux.sh
            bash submit_mjoux.sh
            sleep 600  # Wait 10 minutes before next submission.
        fi
        ctx_len=$((ctx_len * 2))
    done
done
