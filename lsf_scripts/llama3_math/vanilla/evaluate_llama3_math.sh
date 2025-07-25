#!/bin/bash
#BSUB -q cal_h8
#BSUB -P ama02
#BSUB -J evaluate_llama3_math
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=8:j_exclusive=yes"
#BSUB -rn
#BSUB -o lsf_logs/evaluate_llama3/math/vanilla.%J.log
#BSUB -e lsf_logs/evaluate_llama3/math/vanilla.%J.err

module load nvhpc-hpcx-cuda12/24.11
module load singularity/ce_4.2.1
module load openmpi/4.1.7-cuda12.4-ucx1.17.0-c

mkdir -p lsf_logs/evaluate_llama3/math

DATA_PATH=$HOME/projects/llm_adapters/dataset/
DATA_NAME=(AddSub MultiArith SingleEq gsm8k AQuA SVAMP)
MODEL_NAME_OR_PATH=/work/rdc/rdccal/model/meta-llama/Meta-Llama-3-8B
MODEL_NAME=$(echo $MODEL_NAME_OR_PATH | awk -F/ '{print $NF}')
for NAME in "${DATA_NAME[@]}"; do
    echo "evaluate ${MODEL_NAME} with ${NAME}"
    if [ $NAME = "AQuA" ]; then
        singularity run --nv uv-cuda12.sif \
            uv run python experiments/llama3_math/evaluate_math.py \
                --base-name-or-path $MODEL_NAME_OR_PATH \
                --input-json-path ${DATA_PATH}/${NAME}/test.json \
                --output-json-path lsf_logs/evaluate_llama3/math/vanilla_eval_${NAME}.json \
                --is-letter-answer \
                --batch-size 256
    else
        singularity run --nv uv-cuda12.sif \
            uv run python experiments/llama3_math/evaluate_math.py \
                --base-name-or-path $MODEL_NAME_OR_PATH \
                --input-json-path ${DATA_PATH}/${NAME}/test.json \
                --output-json-path lsf_logs/evaluate_llama3/math/vanilla_eval_${NAME}.json \
                --batch-size 256
    fi
done
