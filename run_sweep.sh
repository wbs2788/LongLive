#!/bin/bash

# 定义要运行的步数列表
STEPS=(3000 3500 4000 4900 4950 5000)

# 基础配置文件路径
BASE_CONFIG="configs/longlive_inference.yaml"

# 检查基础配置文件是否存在
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Config file $BASE_CONFIG not found!"
    exit 1
fi

# 遍历每个步数
for STEP in "${STEPS[@]}"; do
    echo "========================================================"
    echo "Preparing inference for Checkpoint Step: $STEP"
    echo "========================================================"

    # 1. 格式化步数，例如 3000 -> 003000 (补齐6位)
    PADDED_STEP=$(printf "%06d" $STEP)
    
    # 2. 定义新的参数
    # 假设 checkpoint 路径格式为 log_new_qa/checkpoint_model_00xxxx/model.pt
    NEW_CKPT="log_new_qa/checkpoint_model_${PADDED_STEP}/model.pt"
    
    # 定义输出文件夹，例如 videos_moviegen/step_3000
    NEW_OUTPUT="videos_moviegen/step_${STEP}"

    # 3. 创建临时配置文件
    TEMP_CONFIG="configs/temp_inference_${STEP}.yaml"
    cp "$BASE_CONFIG" "$TEMP_CONFIG"

    # 4. 使用 sed 替换配置文件中的参数
    # 替换 lora_ckpt (使用 | 作为分隔符以避免路径中的 / 冲突)
    sed -i "s|lora_ckpt: .*|lora_ckpt: ${NEW_CKPT}|g" "$TEMP_CONFIG"
    
    # 替换 output_folder
    sed -i "s|output_folder: .*|output_folder: ${NEW_OUTPUT}|g" "$TEMP_CONFIG"

    echo "Config created: $TEMP_CONFIG"
    echo "  -> LoRA: $NEW_CKPT"
    echo "  -> Output: $NEW_OUTPUT"

    # 5. 运行推理命令
    # 注意：这里使用的是您提供的 inference_hf.py
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
      --nproc_per_node=4 \
      --master_port=29502 \
      inference_hf.py \
      --config_path "$TEMP_CONFIG"

    # 检查上一条命令的退出状态
    if [ $? -eq 0 ]; then
        echo "Successfully finished step $STEP"
        # 成功后删除临时配置文件（可选，如果想保留排查问题可注释掉这行）
        rm "$TEMP_CONFIG"
    else
        echo "Error: Failed at step $STEP"
        exit 1
    fi
    
    echo ""
done

echo "All steps completed!"