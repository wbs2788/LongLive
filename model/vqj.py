from typing import List, Dict, Any, Tuple, Optional
import torch, gc, os
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from pathlib import Path
import json

# =========================
# Video-QA Judge (VQJ) interface
# =========================
# Implement .score(video_rgb[T, 3, H, W], qa_list) -> {"pass_rate": float, "items": ...}
# You can wrap your Qwen-VL inference here.

class VideoQAJudge:
    def score(self, video_rgb: torch.Tensor, qa_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Abstract interface. Implement this to:
          - read video_rgb in [0,1], shape [T, 3, H, W]
          - run your VLM-judge on qa_list
          - return {"pass_rate": float, "items": [...]}
        """
        raise NotImplementedError
    
class JudgeConfig:
    model_path: str = ""
    torch_dtype: str = "bfloat16"
    max_frames: int = 8
    max_new_tokens: int = 32
    yes_patterns: List[str] = None
    no_patterns: List[str] = None
    quantize: str = "4bit"            # "4bit" | "8bit" | "none"
    vision_resize: Tuple[int,int] = (448, 448)  # 下采样到这个分辨率（长边）
    max_memory_cuda_gb: Optional[int] = None    # 例如 20 表示 20GiB
    offload_folder: Optional[str] = None        # CPU卸


class QwenVLJudge(VideoQAJudge):
    def __init__(self, cfg: JudgeConfig):
        assert cfg.model_path and os.path.exists(cfg.model_path), f"Model path not found: {cfg.model_path}"

        self.model_path = model_path
        self.max_frames = int(max_frames)
        self.max_new_tokens = int(max_new_tokens)
        self.vision_resize = vision_resize
        self.yes_patterns = yes_patterns or ["yes", "是", "对", "正确", "存在", "看到了"]
        self.no_patterns  = no_patterns  or ["no", "否", "不对", "错误", "不存在", "没看到"]

        # === bitsandbytes ===
        quant_config = None
        if quantize and quantize.lower() in ("4bit", "8bit"):
            use_4bit = quantize.lower() == "4bit"
            compute_dtype = torch.bfloat16 if torch_dtype in ("bf16","bfloat16") else torch.float16
            quant_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                load_in_8bit=not use_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True if use_4bit else None,
                bnb_4bit_quant_type="nf4" if use_4bit else None,
            )
        
        # === load model & processor ===
        device_map = "auto"
        max_memory = None
        if max_memory_cuda_gb is not None:
            max_memory = { "cpu": "120GiB" }
            cuda_total = torch.cuda.device_count()
            for i in range(cuda_total):
                max_memory[f"cuda:{i}"] = f"{int(max_memory_cuda_gb)}GiB"

        compute_dtype = torch.bfloat16 if torch_dtype in ("bf16","bfloat16") else torch.float16

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2",
            device_map=device_map,
            quantization_config=quant_config,
            max_memory=max_memory,
            offload_folder=offload_folder,
            trust_remote_code=True,
        ).eval()
        print(f"[VQJ] Loaded Qwen-VL model from {model_path}.")
        print(f"[VQJ] Quantization: {quantize}, dtype: {torch_dtype}, max_frames: {max_frames}, max_new_tokens: {max_new_tokens}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.generation_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            use_cache=False,            # 关键：减小KV Cache显存
            do_sample=False,           # 判题更稳定；如需采样可 True 但显存略增
            pad_token_id=getattr(self.model.generation_config, "pad_token_id", None)
        )

    def _ask_consistency(self, frames, question_text: str) -> int:
        """对整段视频问一次一致性，是则 1，否则 0；解析同样走 yes/no 规则。"""
        messages = [{
            "role": "user",
            "content": [
                *({"type": "image", "image": img} for img in frames),
                {"type": "text",
                "text": question_text}
            ],
        }]
        try:
            pred = self._generate_text(messages, force_short=True)
        except Exception:
            return 0
        label = self._to_label(self._norm(pred))
        return 1 if label == "yes" else 0

    @torch.no_grad()
    def score(self, video_rgb: torch.Tensor, qa_list: List[Dict[str, Any]]):
        """
        增强版评分：
        - 少于 min_correct 则直接置 0
        - 追加一致性(binary)问题；默认硬门控：不一致 => 最终 0
        返回：
        {
            "pass_rate": base_pass_rate,         # 原始题目通过率
            "num_correct": int,
            "num_total": int,
            "consistency": int,                  # 1/0
            "final_score": float,                # 结合一致性的最终得分
            "items": [...],                      # 每题细项
        }
        """
        # -------- 参数（可放到 config 里）--------
        min_correct = getattr(self, "min_correct", 5)  # 小于该正确数→置 0
        # 一致性问题文案：你也可以放到 config.grpo.judge.consistency_question
        consistency_question = getattr(
            self, "consistency_question",
            "Considering the entire clip, is the *main subject* consistent across frames? "
            "Answer strictly 'YES' or 'NO'."
        )
        # 一致性整合策略：硬门控 or 软权重
        use_hard_gate = getattr(self, "consis_hard_gate", True)
        # 软权重时：不一致也给一个低权重 alpha（比如 0.2）
        inconsistency_alpha = float(getattr(self, "consis_alpha_if_no", 0.0))
        # 强调全对时的一个可选加分（比如全对就再+5%）
        all_correct_bonus = float(getattr(self, "all_correct_bonus", 0.0))  # 0.0 表示不开

        # -------- 输入检查与取帧 --------
        if not qa_list:
            print("[VQJ][WARN] qa_list is empty")
            return {"pass_rate": 0.0, "items": []}

        frames = self._sample_frames_from_tensor(video_rgb, self.max_frames, self.vision_resize)
        if not frames:
            print("[VQJ][WARN] No frames sampled from video_rgb")
            items = [{"question": qa.get("question",""), "expected": qa.get("answer",""), "predicted": "", "match": 0}
                    for qa in qa_list]
            return {"pass_rate": 0.0, "items": items}

        # -------- 逐题判分 --------
        items = []
        for qa in qa_list:
            q = str(qa.get("question", ""))
            expected_raw = qa.get("answer", "")

            messages = [{
                "role": "user",
                "content": [
                    *({"type": "image", "image": img} for img in frames),
                    {"type": "text",
                    "text": ("Answer strictly 'YES' or 'NO' based only on the images. "
                            "Do not guess beyond what is visible.\n"
                            f"Question: {q}")}
                ],
            }]

            try:
                pred_text = self._generate_text(messages)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._oom_recover()
                    fallback = frames[::2] if len(frames) > 2 else frames[:1]
                    messages = [{
                        "role": "user",
                        "content": [
                            *({"type": "image", "image": img} for img in fallback),
                            {"type": "text",
                            "text": ("Answer strictly 'YES' or 'NO' based only on the images.\n"
                                    f"Question: {q}")}
                        ],
                    }]
                    pred_text = self._generate_text(messages, force_short=True)
                else:
                    raise

            pred_norm  = self._norm(pred_text)
            pred_label = self._to_label(pred_norm)   # 'yes'/'no'/'other'

            exp_list_norm = self._normalize_expected(expected_raw)  # List[str]
            exp_labels    = { self._to_label(x) for x in exp_list_norm }

            # 先按 YES/NO 分类匹配，若期望非二分类再退化到包含/相等
            if ("yes" in exp_labels) or ("no" in exp_labels):
                match = 1 if pred_label in exp_labels else 0
            else:
                match = 1 if any((x == pred_norm) or (x in pred_norm) for x in exp_list_norm) else 0

            items.append({
                "question": q,
                "expected": exp_list_norm,    # 存列表，避免类型坑
                "predicted": pred_text,
                "pred_label": pred_label,
                "match": int(match),
            })

        num_correct = sum(it["match"] for it in items)
        num_total   = len(items)
        base_pass_rate = (num_correct / num_total) if num_total > 0 else 0.0

        # -------- 阈值：少于 K 个正确 → 直接置 0 --------
        if num_correct < min_correct:
            base_pass_rate = 0.0

        # -------- 一致性问题（再问一次）---------
        consis = self._ask_consistency(frames, consistency_question)  # 1/0

        # -------- 全正确加成（可选）---------
        if all_correct_bonus > 0.0 and num_correct == num_total and num_total > 0:
            base_pass_rate = min(1.0, base_pass_rate * (1.0 + all_correct_bonus))

        # -------- 最终得分融合 --------
        if use_hard_gate:
            final_score = base_pass_rate * (1.0 if consis == 1 else 0.0)
        else:
            # 软融合：不一致时给一个很低的权重 alpha（如 0.2）
            c_weight = 1.0 if consis == 1 else float(inconsistency_alpha)
            final_score = base_pass_rate * c_weight

        return {
            "pass_rate": float(base_pass_rate),
            "num_correct": int(num_correct),
            "num_total": int(num_total),
            "consistency": int(consis),
            "final_score": float(final_score),
            "items": items,
        }

    def _generate_text(self, messages, force_short: bool = False) -> str:
        # 编码
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 更短输出用于 OOM 兜底
        gen_kwargs = dict(self.generation_kwargs)
        if force_short:
            gen_kwargs["max_new_tokens"] = min(16, gen_kwargs.get("max_new_tokens", 32))

        # 自动混精度（再省一点激活/临时显存）
        with torch.autocast(device_type="cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # 只取新增 token
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text

    def _oom_recover(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _sample_frames_from_tensor(self, video_rgb: torch.Tensor, max_frames: int,
                                   target_hw: Tuple[int,int]) -> List[Image.Image]:
        if video_rgb is None or video_rgb.numel() == 0:
            return []
        x = video_rgb.detach().float().cpu()
        T = x.shape[0]
        if T == 0:
            return []
        if max_frames <= 0:
            max_frames = 1
        idxs = torch.linspace(0, T - 1, steps=min(T, max_frames)).round().long().tolist()
        Ht, Wt = target_hw
        frames = []
        for i in idxs:
            img = x[i].clamp_(0, 1)                    # [3,H,W]
            img = (img * 255.0).byte().permute(1, 2, 0).numpy()  # [H,W,3]
            pil = Image.fromarray(img)
            if target_hw is not None:
                pil = pil.resize((Wt, Ht), Image.BILINEAR)       
            frames.append(pil)
        return frames
    
    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        for ch in ["。", "！", "!", "？", "?", ".", ",", "，", "、"]:
            s = s.replace(ch, "")
        s = " ".join(s.split())
        return s

    def _to_label(self, s: str) -> str:
        s = self._norm(s)
        yes_set = set(self.yes_patterns + ["y", "true", "1"])
        no_set  = set(self.no_patterns  + ["n", "false", "0"])
        if any(tok == s or tok in s for tok in yes_set):
            return "yes"
        if any(tok == s or tok in s for tok in no_set):
            return "no"
        return "other"

    def _normalize_expected(self, ans) -> list:
        if isinstance(ans, list):
            lst = ans
        else:
            lst = [ans]
        return [self._norm(str(x)) for x in lst if x is not None]
    