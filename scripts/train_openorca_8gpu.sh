#!/usr/bin/env bash
set -euo pipefail

# 数据与超参
DATA="/home/jusheng/yijia/LLaDA/Pretrained/data/OpenOrca/1M-GPT4-Augmented.parquet"
MODEL="/home/jusheng/yijia/LLaDA/Pretrained/model/LLaDA-8B-Instruct"
OUT_DIR="/home/jusheng/yijia/LLaDA/checkpoints_meanflow_llm_openorca"
BATCH_SIZE=${BATCH_SIZE:-1}          # 每卡 batch size
MAX_LEN=${MAX_LEN:-2048}
STEPS=${STEPS:-10000}
PROMPT_LEN=${PROMPT_LEN:-16}
NUM_WORKERS=${NUM_WORKERS:-2}
MAX_EPOCHS=${MAX_EPOCHS:--1}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:--1}
PRECISION=${PRECISION:-bf16-mixed}
ACCUMULATE=${ACCUMULATE:-1}

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

cd /home/jusheng/yijia/LLaDA

export CUDA_VISIBLE_DEVICES=5,6

# 通过 torchrun 启动 8 卡 DDP 训练
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT:-29513} \
  -m trainers.train_openorca_meanflow \
  --parquet "$DATA" \
  --model "$MODEL" \
  --out "$OUT_DIR" \
  --lr ${LR:-1e-4} \
  --prompt_length $PROMPT_LEN \
  --max_len $MAX_LEN \
  --batch_size $BATCH_SIZE \
  --max_steps $STEPS \
  --max_epochs $MAX_EPOCHS \
  --steps_per_epoch $STEPS_PER_EPOCH \
  --precision $PRECISION \
  --accumulate $ACCUMULATE \
  --autograd_jvp \
  --num_workers $NUM_WORKERS

