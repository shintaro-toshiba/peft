#!/bin/bash
#BSUB -q cal_h8
#BSUB -P ama02
#BSUB -J train_llama3_dora_math
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=8:j_exclusive=yes"
#BSUB -rn
#BSUB -o lsf_logs/train_llama3/math/dora.%J.log
#BUSB -o lsf_logs/train_llama3/math/dora.%J.err

module load nvhpc-hpcx-cuda12/24.11
module load singularity/ce_4.2.1
module load openmpi/4.1.7-cuda12.4-ucx1.17.0-c

NUM_GPUS_PER_NODE=8
NUM_NODES=$(cat ${LSB_DJOB_HOSTFILE} | wc -l)
export OMP_NUM_THREADS=1
export MASTER_PORT=29500
export MASTER_ADDR=$(cat ${LSB_DJOB_HOSTFILE} | head -1)
export PYTHONPATH=src

MODEL_NAME_OR_PATH=/work/rdc/rdccal/model/meta-llama/Meta-Llama-3-8B
DATA_PATHS=$HOME/projects/llm_adapters/ft-training_set/math_10k.json
SEED=0
BATCH_SIZE=8
LR=5e-4

MODEL_NAME=$(echo $MODEL_NAME_OR_PATH | awk -F/ '{print $NF}')
singularity run --nv uv-cuda12.sif \
    uv run torchrun --standalone --nproc_per_node=$NUM_GPUS_PER_NODE \
        experiments/llama3_math/train_lora_math.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --json_data_files $DATA_PATHS \
        --seed $SEED \
        --log_level info \
        --bf16 \
        --use_dora \
        --lora_r 32 \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --lora_target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
        --do_train \
        --logging_strategy epoch \
        --save_strategy epoch \
        --warmup_steps 100 \
        --weight_decay 0.0 \
        --max_seq_length 512 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps 1 \
        --learning_rate $LR \
        --num_train_epochs 3 \
        --output_dir hf_models/${MODEL_NAME}/math/dora/seed${SEED}_lr${LR}
