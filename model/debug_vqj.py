# debug_vqj.py
import json, unicodedata
from vqj import JudgeConfig, QwenVLJudge, QwenVLVLLMJudge
import torch

def attach_debug(judge):
    """给 QwenVLJudge / QwenVLVLLMJudge 临时打桩，打印模型回答的详细信息"""
    def _show(tag, s, norm_fn, label_fn):
        norm = norm_fn(s)
        label = label_fn(norm)
        print(f"[{tag}] RAW   :", repr(s))
        print(f"[{tag}] NORM  :", repr(norm))
        print(f"[{tag}] LABEL :", label)
        print(f"[{tag}] CODEP :", " ".join(f"U+{ord(ch):04X}" for ch in s))
        # 仅由标点/空白组成时给出提示（常见 '！！！' 情况）
        only_punct_or_space = all(unicodedata.category(ch).startswith(("P", "Z")) for ch in s) if s else True
        if (not norm) or only_punct_or_space or (label == "other"):
            print(f"[{tag}] WARN  : looks like punctuation-only / normalized-empty → may be judged 'other'")
        print("-" * 56)

    # HF 路径
    if hasattr(judge, "_generate"):
        _orig_gen = judge._generate
        def _gen_debug(messages, force_short=False):
            out = _orig_gen(messages, force_short)
            _show("HF-GEN", out, judge._norm, judge._to_label)
            return out
        judge._generate = _gen_debug

    # vLLM 路径
    if hasattr(judge, "_chat_once"):
        _orig_chat = judge._chat_once
        def _chat_debug(content, max_tokens=4, retries=3, timeout=15.0):
            out = _orig_chat(content, max_tokens, retries, timeout)
            _show("VLLM", out, judge._norm, judge._to_label)
            return out
        judge._chat_once = _chat_debug

    return judge


if __name__ == "__main__":
    # ===== 选一种 Judge 测试 =====
    # 1) HF 直载（需要本地权重）
    # cfg = JudgeConfig(model_path="/path/to/qwen-vl-judge")
    # judge = QwenVLJudge(cfg)

    # 2) vLLM HTTP（需要已起服务）
    cfg = JudgeConfig(model_path="qwen-judge")  # 或你的 served-model 名
    judge = QwenVLVLLMJudge(cfg)

    judge = attach_debug(judge)

    # 做一个极小的测试：1 帧黑图 + 1 道题
    video_rgb = torch.zeros(1, 3, 8, 8)  # [T,3,H,W] in [0,1]，随便占位
    qa_list = [{"question": "Is there any object?", "answer": "NO"}]

    res = judge.score(video_rgb, qa_list)
    print(json.dumps(res, ensure_ascii=False, indent=2))
