IMG="/picassox/intelligent-cpfs/digital-human/intern_digital-human/LongLive/vqj_test_30B_logs/monitor/step_0000700_grpo_chunk/frame_00.png"
OUT="/tmp/payload.json"

# 开头
printf '%s' '{"model":"qwen-judge","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,' > "$OUT"
# 接 base64
base64 -w0 "$IMG" >> "$OUT"
# 收尾
printf '%s' '"}},{"type":"text","text":"Answer YES or NO: is there a cat?"}]}],"max_tokens":4,"temperature":0.0}' >> "$OUT"

# 发送
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  --data-binary @"$OUT" | jq

