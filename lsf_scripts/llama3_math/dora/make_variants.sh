#!/bin/bash

seeds=(0)
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)

output_dir=lsf_scripts/llama3_math/dora/tmp

if [ ! -d ${output_dir} ]; then
    echo "make directory ${output_dir}"
    mkdir -p ${output_dir};
fi

# make new lsf-scripts for each variants.
template_train_file=lsf_scripts/llama3_math/dora/train_llama3_dora_math.sh
template_evaluate_file=lsf_scripts/llama3_math/dora/evaluate_llama3_dora_math.sh
file_name=$(echo $template_train_file | awk -F/ '{print $NF}' | sed 's/\.[^.]*$//')
for seed in "${seeds[@]}"; do
    for lr in "${lrs[@]}"; do
        # make scripts
        output_file="${output_dir}/${file_name}_seed${seed}_lr${lr}.sh"
        cat $template_train_file > $output_file
        tail -n +15 $template_evaluate_file >> $output_file
        sed -i "s/\(.*\).%J.log/\1_seed${seed}_lr${lr}.%J.log/g" "$output_file"
        sed -i "s/\(.*\).%J.err/\1_seed${seed}_lr${lr}.%J.err/g" "$output_file"
        sed -i "s/LR=.*/LR=${lr}/g" "$output_file"
        sed -i "s/SEED=.*/SEED=${seed}/g" "$output_file"
        echo "Generate script: $output_file"
    done
done
