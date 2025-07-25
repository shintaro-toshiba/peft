#!/bin/bash
#BSUB -q cal_h8
#BSUB -P ama02
#BSUB -J train_roberta_bola_stsb
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -rn
#BSUB -o lsf_logs/train_roberta/stsb/bola.%J.log
#BSUB -e lsf_errs/train_roberta/stsb/bola.%J.err

module load nvhpc-hpcx-cuda12/24.11
module load singularity/ce_4.2.1
module load openmpi/4.1.7-cuda12.4-ucx1.17.0-c

mkdir -p lsf_logs/train_roberta/stsb
mkdir -p lsf_errs/train_roberta/stsb

# original results
# https://github.com/microsoft/LoRA/tree/main?tab=readme-ov-file#:~:text=We%20obtain%20result,DeBERTa%20LoRA%20checkpoints.

NUM_GPUS_PER_NODE=1
TASK_NAME=stsb
MODEL_NAME_OR_PATH=/work/rdc/rdccal/model/roberta-base
SEED=0
BATCH_SIZE=64
LR=5e-4

MODEL_NAME=$(echo $MODEL_NAME_OR_PATH | awk -F/ '{print $NF}')
singularity run --nv uv-cuda12.sif \
    uv run torchrun --standalone --nproc_per_node=$NUM_GPUS_PER_NODE \
        experiments/roberta_gleu/train_bola_gleu.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --model_cache_dir hf_models \
        --glue_task_name $TASK_NAME \
        --data_cache_dir hf_datasets \
        --seed $SEED \
        --log_level info \
        --use_bola \
        --bola_num_in_blocks 24 \
        --bola_num_out_blocks 32 \
        --bola_top_k 8 \
        --bola_alpha 6.0 \
        --bola_dropout 0.1 \
        --bola_target_modules query,value \
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
        --metric_for_best_model pearson \
        --greater_is_better True \
        --save_total_limit 3 \
        --output_dir hf_models/bola_${MODEL_NAME}/${TASK_NAME}/seed${SEED}_lr${LR}
