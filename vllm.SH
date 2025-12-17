export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"

CUDA_VISIBLE_DEVICES=3 vllm serve ./Qwen3-VL-30B-A3B-Instruct-FP8 \
  --served-model-name qwen-judge \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --chat-template-content-format openai \
  --limit-mm-per-prompt '{"image": 12}' \
  --max-model-len 16384 \
  --api-key "$VLLM_API_KEY"