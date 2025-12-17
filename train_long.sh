export CUDA_VISIBLE_DEVICES=2,3,4,5

export JUDGE_ENDPOINTS="http://127.0.0.1:8000/v1"
export JUDGE_API_KEY="EMPTY"

torchrun --nproc_per_node=4 train.py \
  --config_path configs/longlive_train_long_vqj_small_vllm.yaml \
  --logdir log_new_qa \
  --wandb-save-dir wandb