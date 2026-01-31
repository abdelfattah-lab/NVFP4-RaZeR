#!/bin/bash

########## Modify the path according to your HOME directory ##########
HOME_DIR="/home/yc2367/llm/NVFP4-RaZeR"
######################################################################

seq_len=2048
OUTPUT_DIR=${HOME_DIR}/results/ppl_${seq_len}
dataset_list="wikitext,c4"

model_list=(
    "llama-2-7b" "llama-2-13b" "llama-3.1-8b" "llama-3.2-3b" \
    "qwen3-4b" "qwen3-8b" "qwen3-14b"
)

w_bits=(4)
w_groupsize=(16)

a_bits=(4)
a_groupsize=(16)

dtype_list=("mxfp4" "nvfp4" "nvfp4_4over6")
w_dtype_list=("nvfp4_razer_e3m3")
a_dtype_list=("nvfp4_razer_e4m3")

for model_name in "${model_list[@]}"
do
    if [[ ${model_name} == "qwen3-8b" ]]
    then
        w_outlier=7.0
    else
        w_outlier=8.0
    fi

    python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
        --datasets ${dataset_list} --seq_len ${seq_len} \
        --output_dir ${OUTPUT_DIR} --use_fp16 \

    for w_dtype in "${w_dtype_list[@]}"
    do
        for a_dtype in "${a_dtype_list[@]}"
        do
            python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
                --datasets ${dataset_list} --seq_len ${seq_len} \
                --output_dir ${OUTPUT_DIR} \
                --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype ${w_dtype} --w_outlier ${w_outlier} \
                --a_bits ${a_bits} --a_groupsize ${a_groupsize} --a_dtype ${a_dtype}
        done
    done

    for dtype in "${dtype_list[@]}"
    do
        python ${HOME_DIR}/run_ppl.py --model_name ${model_name} \
            --datasets ${dataset_list} --seq_len ${seq_len} \
            --output_dir ${OUTPUT_DIR} \
            --w_bits ${w_bits} --w_groupsize ${w_groupsize} --w_dtype ${dtype} --w_outlier ${w_outlier} \
            --a_bits ${a_bits} --a_groupsize ${a_groupsize} --a_dtype ${dtype} 
    done
done


