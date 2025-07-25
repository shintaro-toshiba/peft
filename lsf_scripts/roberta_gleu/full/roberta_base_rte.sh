#!/bin/bash
#BSUB -q cal_h8
#BSUB -P ama02
#BSUB -J train_roberta_full_rte
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=4:j_exclusive=yes"
#BSUB -rn
#BSUB -o lsf_logs/train_roberta/rte/full.%J.log
#BUSB -o lsf_logs/train_roberta/rte/full.%J.err

module load nvhpc-hpcx-cuda12/24.11
module load singularity/ce_4.2.1
module load openmpi/4.1.7-cuda12.4-ucx1.17.0-c

mkdir -p lsf_logs/train_roberta/rte

# original results
# https://github.com/microsoft/LoRA/tree/main?tab=readme-ov-file#:~:text=We%20obtain%20result,DeBERTa%20LoRA%20checkpoints.

NUM_GPUS_PER_NODE=4
TASK_NAME=rte
MODEL_NAME_OR_PATH=/work/rdc/rdccal/model/roberta-base
SEED=0
# reference: https://arxiv.org/pdf/2106.10199
BATCH_SIZE=16
LR=1e-5

MODEL_NAME=$(echo $MODEL_NAME_OR_PATH | awk -F/ '{print $NF}')
singularity run --nv uv-cuda12.sif \
    uv run torchrun --standalone --nproc_per_node=$NUM_GPUS_PER_NODE \
        experiments/roberta_gleu/train_lora_gleu.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --model_cache_dir hf_models \
        --glue_task_name $TASK_NAME \
        --data_cache_dir hf_datasets \
        --seed $SEED \
        --log_level info \
        --do_train \
        --do_eval \
        --eval_strategy epoch \
        --save_strategy epoch \
        --warmup_ratio 0.06 \
        --weight_decay 0.1 \
        --max_seq_length 512 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --num_train_epochs 30 \
        --load_best_model_at_end \
        --metric_for_best_model accuracy \
        --greater_is_better True \
        --save_total_limit 3 \
        --output_dir hf_models/full_${MODEL_NAME}/${TASK_NAME}/seed${SEED}_lr${LR}
