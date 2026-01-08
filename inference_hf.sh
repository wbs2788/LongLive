CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nproc_per_node=4 \
  --master_port=29502 \
  inference_hf.py \
  --config_path configs/longlive_inference.yaml 