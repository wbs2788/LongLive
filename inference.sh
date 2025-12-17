CUDA_VISIBLE_DEVICES=4,5 torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  inference.py \
  --config_path configs/longlive_inference.yaml 