CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  inference.py \
  --config_path configs/longlive_inference.yaml 