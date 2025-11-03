export JUDGE_ENDPOINTS="http://127.0.0.1:8000/v1"
export JUDGE_API_KEY="EMPTY"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 train.py \
  --config_path configs/longlive_train_long_vqj_small_vllm.yaml \
  --logdir vqj_test_30B_logs \
  --wandb-save-dir wandb