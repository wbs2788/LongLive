#!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_long_vqj_small.yaml
LOGDIR=vqj_test_logs
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR