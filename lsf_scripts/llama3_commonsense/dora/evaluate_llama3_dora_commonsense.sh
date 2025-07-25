#!/bin/bash
#BSUB -q cal_h8
#BSUB -P ama02
#BSUB -J evaluate_llama3_dora_commonsense
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=8:j_exclusive=yes"
#BSUB -rn
#BSUB -o lsf_logs/evaluate_llama3/commonsense/dora_eval.%J.log
#BUSB -o lsf_logs/evaluate_llama3/commonsense/dora_eval.%J.err

module load nvhpc-hpcx-cuda12/24.11
module load singularity/ce_4.2.1
module load openmpi/4.1.7-cuda12.4-ucx1.17.0-c

mkdir -p lsf_logs/evaluate_llama3/commonsense

SEED=0
LR=5e-4
DATA_PATH=$HOME/projects/llm_adapters/dataset/
DATA_NAME=(boolq piqa hellaswag ARC-Challenge ARC-Easy openbookqa social_i_qa winogrande)
MODEL_NAME_OR_PATH=/work/rdc/rdccal/model/meta-llama/Meta-Llama-3-8B
PEFT_NAME_OR_PATH=hf_models/${MODEL_NAME}/commonsense/dora/seed${SEED}_lr${LR}
MODEL_NAME=$(echo $PEFT_NAME_OR_PATH | awk -F/ '{print $NF}')

for NAME in "${DATA_NAME[@]}"; do
    echo "evaluate ${MODEL_NAME} with ${NAME}"
    singularity run --nv uv-cuda12.sif \
        uv run python experiments/llama3_commonsense/evaluate_commonsense.py \
            --base-name-or-path $MODEL_NAME_OR_PATH \
            --input-json-path ${DATA_PATH}/${NAME}/test.json \
            --output-json-path lsf_logs/evaluate_llama3/commonsense/vanilla_eval_${NAME}.json \
            --dataname $NAME \
            --batch-size 256
done
