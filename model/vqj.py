# vqj_qwen.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import gc, os, io, base64, time, requests, json
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# =========================
# Video-QA Judge (VQJ) interface
# =========================

class VideoQAJudge:
    def score(self, video_rgb: torch.Tensor, qa_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

# -------- helper --------
def _to_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16"):  return torch.float16
    return torch.float32

def _build_max_memory(max_memory_cuda_gb: Optional[int]) -> Optional[Dict[Union[int,str], str]]:
    if not torch.cuda.is_available() or not max_memory_cuda_gb:
        return None
    d: Dict[Union[int,str], str] = {"cpu": "200GiB"}
    for i in range(torch.cuda.device_count()):
        d[f"cuda:{i}"] = f"{int(max_memory_cuda_gb)}GiB"
    return d

# =========================
# Config
# =========================
@dataclass
class JudgeConfig:
    # 通用
    model_path: str = ""
    backend: str = "hf"
    torch_dtype: str = "bfloat16"
    max_frames: int = 8
    max_new_tokens: int = 32
    yes_patterns: List[str] = field(default_factory=lambda: ["yes", "是", "对", "正确", "存在", "看到了"])
    no_patterns:  List[str] = field(default_factory=lambda: ["no", "否", "不对", "错误", "不存在", "没看到"])
    quantize: str = "4bit"                      # HF 路径使用；vLLM 下忽略
    vision_resize: Tuple[int,int] = (448, 448)  # H, W

    # HF 专用（vLLM 下忽略）
    max_memory_cuda_gb: Optional[int] = None
    offload_folder: Optional[str] = None
    device_map: str = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"

    # 判分策略
    min_correct: int = 0
    consistency_question: str = (
        "Considering the entire clip, is the main subject consistent across frames? "
        "Answer strictly 'YES' or 'NO'."
    )
    consis_hard_gate: bool = False
    consis_alpha_if_no: float = 0.2
    all_correct_bonus: float = 0.0
    scoring_mode: str = "full"

    # ===== vLLM 子配置（新增）=====
    # 若你的 YAML 用 judge.vllm.*，这些值会在 from_config 里被覆盖
    vllm_tensor_parallel_size: int = 8
    vllm_enable_expert_parallel: bool = True
    vllm_kv_cache_dtype: str = "fp8"            # H20/Hopper 建议 fp8；Ampere 用 fp16
    vllm_max_model_len: int = 2048              # 控上下文 → 省 KV
    vllm_gpu_memory_utilization: float = 0.62   # 控 KV 池预留比例（H20 96GB ≈ 60GB/卡）
    vllm_cpu_offload_gb: int = 8                # 需要更省可加大（会变慢）
    vllm_max_num_seqs: int = 8                  # 控并发
    vllm_max_num_batched_tokens: int = 1024     # 控批内总 token
    vllm_enforce_eager: bool = True             # 显存更稳定

    @staticmethod
    def from_config(cfg: Any) -> "JudgeConfig":
        # 兼容 config.grpo.judge / 直接传子配置 / dict 或对象
        def _get(src, key, default=None):
            if isinstance(src, dict):
                return src.get(key, default)
            return getattr(src, key, default)

        src = _get(getattr(cfg, "grpo", None), "judge", None) or cfg
        out = JudgeConfig(
            model_path=_get(src, "model_path", ""),
            backend=getattr(src, "backend", "hf") if not isinstance(src, dict) else src.get("backend", "hf"),
            torch_dtype=_get(src, "torch_dtype", "bfloat16"),
            max_frames=int(_get(src, "max_frames", 8) or 8),
            max_new_tokens=int(_get(src, "max_new_tokens", 32) or 32),
            yes_patterns=list(_get(src, "yes_patterns", ["yes","是","对","正确","存在","看到了"])),
            no_patterns=list(_get(src, "no_patterns", ["no","否","不对","错误","不存在","没看到"])),
            quantize=_get(src, "quantize", "4bit"),
            vision_resize=tuple(_get(src, "vision_resize", (448,448))),
            max_memory_cuda_gb=_get(src, "max_memory_cuda_gb", None),
            offload_folder=_get(src, "offload_folder", None),
            device_map=_get(src, "device_map", "auto"),
            attn_implementation=_get(src, "attn_implementation", "flash_attention_2"),
            min_correct=int(_get(src, "min_correct", 0) or 0),
            consistency_question=_get(
                src, "consistency_question",
                "Considering the entire clip, is the main subject consistent across frames? Answer strictly 'YES' or 'NO'."
            ),
            consis_hard_gate=bool(_get(src, "consis_hard_gate", False)),
            consis_alpha_if_no=float(_get(src, "consis_alpha_if_no", 0.2) or 0.2),
            all_correct_bonus=float(_get(src, "all_correct_bonus", 0.0) or 0.0),
            scoring_mode=_get(src, "scoring_mode", "full"),
        )

        # 读取 judge.vllm.*（若存在则覆盖）
        vllm_src = _get(src, "vllm", {}) or {}
        def _ovr(name, attr):
            val = _get(vllm_src, name, None)
            if val is not None:
                setattr(out, attr, val)

        _ovr("tensor_parallel_size", "vllm_tensor_parallel_size")
        _ovr("enable_expert_parallel", "vllm_enable_expert_parallel")
        _ovr("kv_cache_dtype", "vllm_kv_cache_dtype")
        _ovr("max_model_len", "vllm_max_model_len")
        _ovr("gpu_memory_utilization", "vllm_gpu_memory_utilization")
        _ovr("cpu_offload_gb", "vllm_cpu_offload_gb")
        _ovr("max_num_seqs", "vllm_max_num_seqs")
        _ovr("max_num_batched_tokens", "vllm_max_num_batched_tokens")
        _ovr("enforce_eager", "vllm_enforce_eager")

        # 也支持扁平命名（如果有人直接写在 judge.* 下）
        _ovr("vllm_tensor_parallel_size", "vllm_tensor_parallel_size")
        _ovr("vllm_enable_expert_parallel", "vllm_enable_expert_parallel")
        _ovr("vllm_kv_cache_dtype", "vllm_kv_cache_dtype")
        _ovr("vllm_max_model_len", "vllm_max_model_len")
        _ovr("vllm_gpu_memory_utilization", "vllm_gpu_memory_utilization")
        _ovr("vllm_cpu_offload_gb", "vllm_cpu_offload_gb")
        _ovr("vllm_max_num_seqs", "vllm_max_num_seqs")
        _ovr("vllm_max_num_batched_tokens", "vllm_max_num_batched_tokens")
        _ovr("vllm_enforce_eager", "vllm_enforce_eager")

        return out

# =========================
# Qwen-VL Judge
# =========================
class QwenVLJudge(VideoQAJudge):
    def __init__(self, cfg: JudgeConfig):
        assert cfg.model_path and os.path.exists(cfg.model_path), f"Model path not found: {cfg.model_path}"
        self.cfg = cfg

        # --- quantization ---
        quant_config = None
        q = (cfg.quantize or "none").lower()
        compute_dtype = _to_dtype(cfg.torch_dtype)
        if q == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif q == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # --- memory & device map ---
        max_memory = _build_max_memory(cfg.max_memory_cuda_gb)

        # --- load model & processor ---
        kwargs = dict(
            torch_dtype=compute_dtype,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        if cfg.offload_folder:
            kwargs["offload_folder"] = cfg.offload_folder

        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_path, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)

        pad_id = getattr(self.model.generation_config, "pad_token_id", None)
        eos_id = getattr(self.model.generation_config, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            pad_id = eos_id  # 避免部分权重没有 pad_id 导致 generate 警告

        self.generation_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            use_cache=False,         # 节省 KV cache 显存
            do_sample=False,
            pad_token_id=pad_id,
        )

        # yes/no 词表
        self.yes_patterns = set(cfg.yes_patterns + ["y", "true", "1"])
        self.no_patterns  = set(cfg.no_patterns  + ["n", "false", "0"])

    # ---------- public ----------
    @torch.no_grad()
    def score(self, video_rgb: torch.Tensor, qa_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        mode = (getattr(self.cfg, "scoring_mode", "full") or "full").lower()

        if video_rgb is not None and video_rgb.numel() > 0:
            vid_t = video_rgb.detach().float()
            if vid_t.max() > 1.0:
                vid_t = vid_t / 255.0
            
            if vid_t.shape[-1] == 3: # [T, H, W, C] -> [T, C, H, W]
                vid_t = vid_t.permute(0, 3, 1, 2)
                
            diffs = torch.abs(vid_t[1:] - vid_t[:-1])
            motion_score = diffs.mean().item()
        else:
            motion_score = 0.0
        

        frames = self._sample_frames(video_rgb, self.cfg.max_frames, self.cfg.vision_resize)
        if not frames:
            items = [{"question": qa.get("question",""), "expected": qa.get("answer",""),
                  "predicted": "", "match": 0} for qa in (qa_list or [])] if mode != "consistency_only" else []
            return {
                "mode": mode,
                "pass_rate": 0.0,
                "num_correct": 0,
                "num_total": len(items),
                "consistency": 0,
                "final_score": 0.0,
                "items": items,
            }

        items = []
        num_correct = 0
        num_total = 0
        base_pass = 1.0

        if mode != "consistency_only":
            if not qa_list:
                return {
                    "mode": mode,
                    "pass_rate": 0.0,
                    "num_correct": 0,
                    "num_total": 0,
                    "consistency": 0 if mode != "qa_only" else 1, 
                    "final_score": 0.0 if mode != "consistency_only" else 0.0,
                    "items": [],
                }

            for qa in qa_list:
                q = str(qa.get("question", ""))
                expected_raw = qa.get("answer", "")

                messages = [{
                    "role": "user",
                    "content": [*({"type":"image","image":img} for img in frames),
                                {"type":"text","text":
                                    ("Answer strictly 'YES' or 'NO' based only on the images.\n"
                                    f"Question: {q}")}],
                }]

                try:
                    pred_text = self._generate(messages)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self._oom_recover()
                        fallback = frames[::2] if len(frames) > 2 else frames[:1]
                        messages[0]["content"] = [*({"type":"image","image":img} for img in fallback),
                                                {"type":"text","text":
                                                ("Answer strictly 'YES' or 'NO' based only on the images.\n"
                                                    f"Question: {q}")}]
                        pred_text = self._generate(messages, force_short=True)
                    else:
                        raise

                pred_norm  = self._norm(pred_text)
                pred_label = self._to_label(pred_norm)
                exp_list   = self._normalize_expected(expected_raw)
                exp_labels = { self._to_label(x) for x in exp_list }

                if ("yes" in exp_labels) or ("no" in exp_labels):
                    match = 1 if pred_label in exp_labels else 0
                else:
                    match = 1 if any((x == pred_norm) or (x in pred_norm) for x in exp_list) else 0

                items.append({
                    "question": q,
                    "expected": exp_list,
                    "predicted": pred_text,
                    "pred_label": pred_label,
                    "match": int(match),
                })

            num_correct = sum(it["match"] for it in items)
            num_total   = len(items)
            base_pass   = (num_correct / num_total) if num_total > 0 else 0.0

            if self.cfg.min_correct and num_total > 0 and num_correct < self.cfg.min_correct:
                base_pass = 0.0

            if self.cfg.all_correct_bonus > 0.0 and num_total > 0 and num_correct == num_total:
                base_pass = min(1.0, base_pass * (1.0 + self.cfg.all_correct_bonus))

        consis = 1.0  # 对于 qa_only，我们把一致性视为中性 1（不影响最终分）
        motion_factor = 1.0
        if mode != "qa_only":
            consis = self._ask_consistency(frames, self.cfg.consistency_question)
        
        REF_MOTION = 0.02
        motion_factor = (motion_score / REF_MOTION) ** 2
        motion_factor = max(0.1, min(motion_factor, 3.0))

        if mode == "qa_only":
            final_score = base_pass * motion_factor # 加上惩罚
            pass_rate   = base_pass
        elif mode == "consistency_only":
            final_score = float(consis) * motion_factor # 加上惩罚
            pass_rate   = float(consis)
        else:
            # full mode
            if self.cfg.consis_hard_gate:
                final_score = base_pass * (1.0 if consis == 1 else 0.0)
            else:
                alpha = 1.0 if consis == 1 else float(self.cfg.consis_alpha_if_no)
                final_score = base_pass * alpha
            pass_rate = base_pass
            # ★★★ 最终应用动态惩罚 ★★★
            final_score = final_score * motion_factor

        return {
            "mode": mode,
            "pass_rate": float(pass_rate),
            "num_correct": int(num_correct),
            "num_total": int(num_total),
            "consistency": int(consis),
            "final_score": float(final_score),
            "motion_score": motion_score, 
            "motion_factor": motion_factor, 
            "items": items,
        }

    @torch.no_grad()
    def rewrite_teacher_prompt(
        self,
        base_prompt: str,
        qa_items: List[Dict[str, Any]],
        max_new_tokens: int = 128,
    ) -> str:
        """
        根据没有 match 的 QA，改写一条更强约束的 prompt，给 teacher 模型用。
        - base_prompt: 原始的视频生成指令
        - qa_items: vqj.score(...) 里返回的 items（其中有 question / expected / match）
        - lang: "zh" 或 "en"，决定提示用中文还是英文描述
        """
        # 1) 只取没通过的 QA
        wrong_items = [it for it in (qa_items or []) if not it.get("match")]
        if not wrong_items:
            # 没有错误就直接用原 prompt
            return base_prompt

        # 2) 把错误 QA 列成文本（问题 + 期望答案）
        lines = []
        for i, it in enumerate(wrong_items, 1):
            q = str(it.get("question", "")).strip()
            # expected 是归一化后的 list[str]
            exp_list = it.get("expected") or []
            if isinstance(exp_list, str):
                exp_list = [exp_list]
            exp_str = " / ".join(exp_list) if exp_list else ""
            line = f"{i}. Q: {q}\n   Expected answer: {exp_str}"
            lines.append(line)
        qa_block = "\n".join(lines)

        # 3) 构造让 Qwen 改写的 meta-prompt
        prompt_text = (
            "You are a video prompt refinement assistant.\n\n"
            "Goal:\n"
            "Refine the original prompt to include missing visual details (based on QA) and ensure dynamic motion, while STRICTLY preserving the original content.\n\n"
            "Requirements:\n"
            "1. **STRICT PRESERVATION**: You must keep the original prompt's structure, vocabulary, and artistic style EXACTLY as is. Do NOT rephrase, summarize, or remove any existing words. Your job is to *insert* missing details, not to rewrite the story.\n"
            "2. **Targeted Injection**: For each 'Q / Expected' item, seamless INSERT short, specific visual keywords into the original text to satisfy the condition. Do not alter the surrounding context.\n"
            "3. **Force Dynamics**: Ensure the prompt explicitly describes MOTION. If the original prompt is static, you MUST append a motion descriptor (e.g., 'slow motion', 'dynamic camera pan', 'moving subject') that fits the scene. Do not let the video be static.\n"
            "4. Output only ONE final prompt, without explanations.\n\n"
            f"Original prompt:\n{base_prompt}\n\n"
            "Visual QA items to fix:\n"
            f"{qa_block}\n\n"
            "Refined prompt:"
        )

        # 4) 调用 Qwen-VL 做纯文本生成
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }]

        # 这里单独指定 max_new_tokens，避免影响 VQA 生成配置
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens

        cm = torch.autocast("cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad()
        with cm:
            out = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        new_prompt = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        if not new_prompt:
            print(f"Warning! Rewrite model is now unreachable.")

        force_suffix = ", high dynamic motion, cinematography, action shot, 4k"
        if not new_prompt.endswith(force_suffix):
            new_prompt += force_suffix
            
        return new_prompt or base_prompt
    
    def rewrite_teacher_prompts_pair(
        self,
        prompt_a: str,
        prompt_b: str,
        qa_items: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
    ) -> Tuple[str, str]:

        wrong_items = [it for it in (qa_items or []) if not it.get("match")]
        if not wrong_items:
            return prompt_a, prompt_b

        # 构造 QA 文本块
        lines = []
        for i, it in enumerate(wrong_items, 1):
            q = str(it.get("question", "")).strip()
            exp_list = it.get("expected") or []
            if isinstance(exp_list, str):
                exp_list = [exp_list]
            exp_str = " / ".join(exp_list)
            line = f"{i}. Q: {q}\n   Expected answer: {exp_str}"
            lines.append(line)
        qa_block = "\n".join(lines)

        prompt_text = (
            "You are a video prompt rewriting assistant.\n\n"
            "The video has TWO segments in time order:\n"
            " - Prompt A describes the EARLY part of the video.\n"
            " - Prompt B describes the LATER part of the video after a transition.\n\n"
            "Goal:\n"
            "Given the original Prompt A and Prompt B, and several visual QA items that "
            "are currently answered incorrectly, rewrite BOTH prompts so that the desired "
            "visual conditions become explicit.\n\n"
            "Requirements:\n"
            "1. Preserve the main story, characters and setting of both prompts as much as possible.\n"
            "2. For each \"Q / Expected\", decide whether it refers mainly to the early segment (A), "
            "the later segment (B), or both. Then inject the expected visual condition into the appropriate prompt.\n"
            "3. Keep the temporal structure: A happens before B.\n"
            "4. Do NOT mention words like 'QA', 'evaluation', 'model', 'score'; "
            "only describe what should appear in the video.\n"
            "5. Output in the following format ONLY:\n"
            "   Prompt A: <rewritten prompt A on one line>\n"
            "   Prompt B: <rewritten prompt B on one line>\n\n"
            "For EVERY QA item listed below:\n"
            "- If the expected answer is 'yes':\n"
            "You MUST ensure that the described event HAPPENS in the video, and you MUST explicitly describe this event in the rewritten prompt.\n"
            "- If the expected answer is 'no':\n"
            "You MUST ensure that this event does NOT happen in the video, and you MUST make the absence clear in the prompt if needed.\n"
            "Do not skip any QA. Every QA must influence the rewritten prompt.\n\n"
            f"Original Prompt A:\n{prompt_a}\n\n"
            f"Original Prompt B:\n{prompt_b}\n\n"
            "These visual QA items were answered incorrectly by the current video:\n"
            f"{qa_block}\n\n"
            "Now provide the rewritten prompts:\n"
            "Prompt A:"
        )

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens

        cm = torch.autocast("cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad()
        with cm:
            out = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        full_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # 简单解析 Prompt A / B
        new_a, new_b = prompt_a, prompt_b
        if "Prompt B:" in full_text:
            parts = full_text.split("Prompt B:", 1)
            a_part = parts[0].replace("Prompt A:", "").strip()
            b_part = parts[1].strip()
            if a_part:
                new_a = a_part
            if b_part:
                new_b = b_part
        else:
            # 万一只返回了一段，就默认当 A 用
            if full_text:
                new_a = full_text

        return new_a, new_b


    # ---------- internals ----------
    def _generate(self, messages, force_short: bool = False) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = dict(self.generation_kwargs)
        if force_short:
            gen_kwargs["max_new_tokens"] = min(16, gen_kwargs.get("max_new_tokens", 32))

        cm = torch.autocast("cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad()
        with cm:
            out = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text

    def _ask_consistency(self, frames: List[Image.Image], q: str) -> int:
        messages = [{
            "role": "user",
            "content": [*({"type":"image","image":img} for img in frames),
                        {"type":"text","text": q}],
        }]
        try:
            pred = self._generate(messages, force_short=True)
        except Exception:
            return 0
        return 1 if self._to_label(self._norm(pred)) == "yes" else 0

    def _oom_recover(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _sample_frames(self, video_rgb: torch.Tensor, max_frames: int, target_hw: Tuple[int,int]) -> List[Image.Image]:
        if video_rgb is None or video_rgb.numel() == 0:
            return []
        x = video_rgb.detach().float().cpu()  # [T,3,H,W] in [0,1]
        T = int(x.shape[0])
        if T <= 0:
            return []
        K = max(1, min(T, int(max_frames)))
        idxs = torch.linspace(0, T - 1, steps=K).round().long().tolist()
        Ht, Wt = target_hw
        frames: List[Image.Image] = []
        for i in idxs:
            img = x[i].clamp_(0, 1)
            img = (img * 255).byte().permute(1, 2, 0).numpy()  # [H,W,3]
            pil = Image.fromarray(img, mode="RGB").resize((Wt, Ht), Image.BILINEAR)
            frames.append(pil)
        return frames

    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        for ch in ["。","！","!","？","?","。","，",",","、"]:
            s = s.replace(ch, "")
        s = " ".join(s.split())
        return s

    def _to_label(self, s: str) -> str:
        # 朴素 token 匹配（已归一化）
        if any(tok == s or tok in s for tok in self.yes_patterns):
            return "yes"
        if any(tok == s or tok in s for tok in self.no_patterns):
            return "no"
        return "other"

    def _normalize_expected(self, ans: Any) -> List[str]:
        if isinstance(ans, (list, tuple)):
            lst = list(ans)
        else:
            lst = [ans]
        return [self._norm(str(x)) for x in lst if x is not None]

class QwenVLVLLMJudge(VideoQAJudge):
    """
    使用 vLLM server 的 OpenAI 兼容 HTTP 接口进行判分。
    要求服务端用：
      --trust-remote-code
      --chat-template-content-format openai-multimodal
      --served-model-name qwen-judge   # 建议
    """
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.model = (cfg.model_path or "qwen-judge").strip()

        # === 必须最先初始化：YES/NO 词表（调试桩/判别逻辑会用到）===
        def _norm_list(xs):
            return [str(x).strip().lower() for x in (xs or []) if x is not None]
        self.yes_patterns = set(_norm_list(cfg.yes_patterns) + ["yes", "y", "true", "1", "是", "对", "正确", "存在", "看到了"])
        self.no_patterns  = set(_norm_list(cfg.no_patterns)  + ["no",  "n", "false","0", "否", "不对","错误","不存在","没看到"])

        # === base_url / session ===
        raw_base = getattr(cfg, "vllm_base_url", None) or \
                   os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.base_url = self._normalize_base_url(raw_base)
        self.api_key  = getattr(cfg, "vllm_api_key", None) or \
                        os.environ.get("VLLM_API_KEY", "EMPTY")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        # 轻量级重试，缓解瞬时 5xx/429 抖动
        retry = Retry(
            total=2, connect=2, read=2, backoff_factor=0.2,
            status_forcelist=[408, 425, 429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"])
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

        # 启动时探活 & 对齐模型名
        self._ensure_model_name()

    @staticmethod
    def _normalize_base_url(u: str) -> str:
        u = (u or "").strip().rstrip("/")
        return (u + "/v1") if not u.endswith("/v1") else u
    
    def _ensure_model_name(self):
        url = f"{self.base_url}/models"
        r = self.session.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        names = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        if not names:
            raise RuntimeError("vLLM /v1/models 返回为空（检查服务端启动与鉴权）")
        if self.model not in names:
            if len(names) == 1:
                self.model = names[0]  # 单模型：自动对齐
            else:
                raise RuntimeError(f"模型名不匹配：self.model='{self.model}'；可用={names}")
            
    # ---------- public ----------
    @torch.no_grad()
    def score(self, video_rgb: torch.Tensor, qa_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        mode = (getattr(self.cfg, "scoring_mode", "full") or "full").lower()
        frames = self._sample_frames(video_rgb, self.cfg.max_frames, self.cfg.vision_resize)
        if not frames:
            items = [{"question": qa.get("question",""), "expected": qa.get("answer",""),
                      "predicted": "", "match": 0} for qa in (qa_list or [])] if mode != "consistency_only" else []
            return {"mode": mode, "pass_rate": 0.0, "num_correct": 0, "num_total": len(items),
                    "consistency": 0, "final_score": 0.0, "items": items}

        items, num_correct, base_pass = [], 0, 1.0

        # ---- QA 判分 ----
        if mode != "consistency_only":
            if not qa_list:
                return {"mode": mode, "pass_rate": 0.0, "num_correct": 0, "num_total": 0,
                        "consistency": 0 if mode != "qa_only" else 1, "final_score": 0.0, "items": []}

            for qa in qa_list:
                q = str(qa.get("question", ""))
                expected_raw = qa.get("answer", "")

                content = [*({"type":"image_url", "image_url":{"url": self._pil_to_dataurl(img)}} for img in frames),
                           {"type":"text","text":"Answer strictly 'YES' or 'NO' based only on the images, and then add the reason.\n"
                                                 f"Question: {q}"}]
                pred_text = self._chat_yesno(frames, q, max_tokens=min(self.cfg.max_new_tokens, 4))

                pred_norm  = self._norm(pred_text)
                pred_label = self._to_label(pred_norm)
                exp_list   = self._normalize_expected(expected_raw)
                exp_labels = { self._to_label(x) for x in exp_list }

                if ("yes" in exp_labels) or ("no" in exp_labels):
                    match = 1 if pred_label in exp_labels else 0
                else:
                    match = 1 if any((x == pred_norm) or (x in pred_norm) for x in exp_list) else 0

                items.append({
                    "question": q,
                    "expected": exp_list,
                    "predicted": pred_text,
                    "pred_label": pred_label,
                    "match": int(match),
                })

            num_correct = sum(it["match"] for it in items)
            num_total   = len(items)
            base_pass   = (num_correct / num_total) if num_total > 0 else 0.0

            if self.cfg.min_correct and num_total > 0 and num_correct < self.cfg.min_correct:
                base_pass = 0.0
            if self.cfg.all_correct_bonus > 0.0 and num_total > 0 and num_correct == num_total:
                base_pass = min(1.0, base_pass * (1.0 + self.cfg.all_correct_bonus))

        # ---- 一致性判定（qa_only 跳过） ----
        consis = 1
        if mode != "qa_only":
            consis = self._ask_consistency_http(frames, self.cfg.consistency_question)

        # ---- 组合得分 ----
        if mode == "qa_only":
            final_score = base_pass
            pass_rate   = base_pass
        elif mode == "consistency_only":
            final_score = float(consis)
            pass_rate   = float(consis)
            items, num_correct, num_total = [], 0, 0
        else:
            if self.cfg.consis_hard_gate:
                final_score = base_pass * (1.0 if consis == 1 else 0.0)
            else:
                alpha = 1.0 if consis == 1 else float(self.cfg.consis_alpha_if_no)
                final_score = base_pass * alpha
            pass_rate = base_pass

        return {
            "mode": mode,
            "pass_rate": float(pass_rate),
            "num_correct": int(num_correct),
            "num_total": int(len(items)),
            "consistency": int(consis),
            "final_score": float(final_score),
            "items": items,
        }

            # ---------- rewrite: 根据错误的 QA 改写 teacher prompt ----------
    @torch.no_grad()
    def rewrite_teacher_prompt(
        self,
        base_prompt: str,
        qa_items: List[Dict[str, Any]],
        max_new_tokens: int = 128,
    ) -> str:
        """
        根据没有 match 的 QA，改写一条更强约束的 prompt，给 teacher 模型用。
        - base_prompt: 原始的视频生成指令
        - qa_items: vqj.score(...) 里返回的 items（其中有 question / expected / match）
        - lang: "zh" 或 "en"，决定提示用中文还是英文描述
        """
        # 1) 只取没通过的 QA
        wrong_items = [it for it in (qa_items or []) if not it.get("match")]
        if not wrong_items:
            # 没有错误就直接用原 prompt
            return base_prompt

        # 2) 把错误 QA 列成文本（问题 + 期望答案）
        lines = []
        for i, it in enumerate(wrong_items, 1):
            q = str(it.get("question", "")).strip()
            # expected 是归一化后的 list[str]
            exp_list = it.get("expected") or []
            if isinstance(exp_list, str):
                exp_list = [exp_list]
            exp_str = " / ".join(exp_list) if exp_list else ""
            line = f"{i}. Q: {q}\n   Expected answer: {exp_str}"
            lines.append(line)
        qa_block = "\n".join(lines)

        # 3) 构造让 Qwen 改写的 meta-prompt
        prompt_text = (
            "You are a video prompt rewriting assistant.\n\n"
            "Goal:\n"
            "Given an original video generation instruction and several visual QA items "
            "that are currently answered incorrectly, rewrite the instruction so that "
            "the desired visual conditions become explicit.\n\n"
            "Requirements:\n"
            "CRITICAL: You MUST strictly PRESERVE all visual details describing the BACKGROUND, SCENE, lighting, and environment from the original prompt. Do NOT delete any scene keywords.\n"
            "1. Preserve the main story, characters and setting of the original prompt as much as possible.\n"
            "2. For each \"Q / Expected\", turn the expected answer into a clear visual condition "
            "and explicitly incorporate it into the new prompt.\n"
            "3. Do NOT mention words like 'QA', 'evaluation', 'model', 'score', etc.; "
            "only describe what should appear in the video.\n"
            "4. Output only ONE rewritten prompt, without explanations.\n\n"
            f"Original prompt:\n{base_prompt}\n\n"
            "These visual QA items were answered incorrectly by the current video:\n"
            f"{qa_block}\n\n"
            "Now provide the rewritten prompt:"
        )

        # 4) 调用 Qwen-VL 做纯文本生成
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }]

        # 这里单独指定 max_new_tokens，避免影响 VQA 生成配置
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens

        cm = torch.autocast("cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad()
        with cm:
            out = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        new_prompt = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        if not new_prompt:
            print(f"Warning! Rewrite model is now unreachable.")
        return new_prompt or base_prompt
    
    def rewrite_teacher_prompts_pair(
        self,
        prompt_a: str,
        prompt_b: str,
        qa_items: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
    ) -> Tuple[str, str]:

        wrong_items = [it for it in (qa_items or []) if not it.get("match")]
        if not wrong_items:
            return prompt_a, prompt_b

        # 构造 QA 文本块
        lines = []
        for i, it in enumerate(wrong_items, 1):
            q = str(it.get("question", "")).strip()
            exp_list = it.get("expected") or []
            if isinstance(exp_list, str):
                exp_list = [exp_list]
            exp_str = " / ".join(exp_list)
            line = f"{i}. Q: {q}\n   Expected answer: {exp_str}"
            lines.append(line)
        qa_block = "\n".join(lines)

        prompt_text = (
            "You are a video prompt rewriting assistant.\n\n"
            "The video has TWO segments in time order:\n"
            " - Prompt A describes the EARLY part of the video.\n"
            " - Prompt B describes the LATER part of the video after a transition.\n\n"
            "Goal:\n"
            "Given the original Prompt A and Prompt B, and several visual QA items that "
            "are currently answered incorrectly, rewrite BOTH prompts so that the desired "
            "visual conditions become explicit.\n\n"
            "Requirements:\n"
            "1. Preserve the main story, characters and setting of both prompts as much as possible.\n"
            "2. For each \"Q / Expected\", decide whether it refers mainly to the early segment (A), "
            "the later segment (B), or both. Then inject the expected visual condition into the appropriate prompt.\n"
            "3. Keep the temporal structure: A happens before B.\n"
            "4. Do NOT mention words like 'QA', 'evaluation', 'model', 'score'; "
            "only describe what should appear in the video.\n"
            "5. Output in the following format ONLY:\n"
            "   Prompt A: <rewritten prompt A on one line>\n"
            "   Prompt B: <rewritten prompt B on one line>\n\n"
            "For EVERY QA item listed below:\n"
            "- If the expected answer is 'yes':\n"
            "You MUST ensure that the described event HAPPENS in the video, and you MUST explicitly describe this event in the rewritten prompt.\n"
            "- If the expected answer is 'no':\n"
            "You MUST ensure that this event does NOT happen in the video, and you MUST make the absence clear in the prompt if needed.\n"
            "Do not skip any QA. Every QA must influence the rewritten prompt.\n\n"
            f"Original Prompt A:\n{prompt_a}\n\n"
            f"Original Prompt B:\n{prompt_b}\n\n"
            "These visual QA items were answered incorrectly by the current video:\n"
            f"{qa_block}\n\n"
            "Now provide the rewritten prompts:\n"
            "Prompt A:"
        )

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["max_new_tokens"] = max_new_tokens

        cm = torch.autocast("cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad()
        with cm:
            out = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        full_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # 简单解析 Prompt A / B
        new_a, new_b = prompt_a, prompt_b
        if "Prompt B:" in full_text:
            parts = full_text.split("Prompt B:", 1)
            a_part = parts[0].replace("Prompt A:", "").strip()
            b_part = parts[1].strip()
            if a_part:
                new_a = a_part
            if b_part:
                new_b = b_part
        else:
            # 万一只返回了一段，就默认当 A 用
            if full_text:
                new_a = full_text

        return new_a, new_b

    # ---------- internals ----------
    def _ensure_model_name(self):
        """探活 /v1/models，校验/纠正 self.model。"""
        url = f"{self.base_url.rstrip('/')}/models"
        try:
            r = self.session.get(url, timeout=5)
            r.raise_for_status()
            names = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
            if not names:
                raise RuntimeError("vLLM /v1/models 返回为空")
            if self.model not in names:
                # 如果没找到，且只有一个模型，就自动用它；否则报错提示
                if len(names) == 1:
                    self.model = names[0]
                else:
                    raise RuntimeError(f"模型名不匹配：当前 self.model='{self.model}'；可用={names}")
        except Exception as e:
            raise RuntimeError(f"无法访问 {url}（确保带了 Authorization 头，服务端起了 --api-key 或去掉鉴权）：{e}")

    def _chat_once(self, content, max_tokens=4, retries=3, timeout=15.0) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
        }

        def _post():
            return self.session.post(url, json=payload, timeout=timeout)

        last_err = None
        tried_refresh = False
        for _ in range(max(1, int(retries))):
            try:
                r = _post()
                if r.status_code == 200:
                    data = r.json()
                    return (data["choices"][0]["message"]["content"] or "").strip()

                # 401 → 明确提示鉴权
                if r.status_code == 401:
                    raise RuntimeError("401 Unauthorized：请确认客户端 Authorization 头与服务端 --api-key 一致")

                # 404 → 尝试自愈（模型名不匹配/服务刚重启）
                if r.status_code == 404 and (("does not exist" in r.text) or ("model" in r.text.lower())):
                    if not tried_refresh:
                        tried_refresh = True
                        self._ensure_model_name()       # 重新发现模型名
                        payload["model"] = self.model   # 替换后重试一次
                        time.sleep(0.05)
                        continue

                # 其它 4xx/5xx
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            except requests.RequestException as e:
                last_err = e

            time.sleep(0.2)

        # 失败兜底
        return ""
    
    def _chat_yesno(self, frames, question, max_tokens=8):
        content = [*({"type":"image_url","image_url":{"url": self._pil_to_dataurl(img)}} for img in frames),
                {"type":"text","text": f"Answer strictly 'YES' or 'NO' based only on the images.\nQuestion: {question}"}]
        txt = self._chat_once(content, max_tokens=max_tokens)
        n = self._norm(txt)
        if (not n) or self._to_label(n) == "other":
            content[-1]["text"] += "\nOnly reply YES or NO (uppercase). No punctuation."
            txt = self._chat_once(content, max_tokens=min(3, max_tokens))
        return txt

    def _ask_consistency_http(self, frames: List[Image.Image], q: str) -> int:
        content = [*({"type":"image_url","image_url":{"url": self._pil_to_dataurl(img)}} for img in frames),
                   {"type":"text","text": q}]
        pred = self._chat_once(content, max_tokens=4)
        return 1 if self._to_label(self._norm(pred)) == "yes" else 0

    def _pil_to_dataurl(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)  # 统一转JPEG，减小体积
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    
    def _generate(self, messages, force_short: bool = False) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_kwargs = dict(self.generation_kwargs)
        if force_short:
            gen_kwargs["max_new_tokens"] = min(16, gen_kwargs.get("max_new_tokens", 32))

        cm = torch.autocast("cuda", dtype=self.model.dtype) if torch.cuda.is_available() else torch.no_grad()
        with cm:
            out = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text

    def _ask_consistency(self, frames: List[Image.Image], q: str) -> int:
        messages = [{
            "role": "user",
            "content": [*({"type":"image","image":img} for img in frames),
                        {"type":"text","text": q}],
        }]
        try:
            pred = self._generate(messages, force_short=True)
        except Exception:
            return 0
        return 1 if self._to_label(self._norm(pred)) == "yes" else 0

    def _oom_recover(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _sample_frames(self, video_rgb: torch.Tensor, max_frames: int, target_hw: Tuple[int,int]) -> List[Image.Image]:
        if video_rgb is None or video_rgb.numel() == 0:
            return []
        x = video_rgb.detach().float().cpu()  # [T,3,H,W] in [0,1]
        T = int(x.shape[0])
        if T <= 0:
            return []
        K = max(1, min(T, int(max_frames)))
        idxs = torch.linspace(0, T - 1, steps=K).round().long().tolist()
        Ht, Wt = target_hw
        frames: List[Image.Image] = []
        for i in idxs:
            img = x[i].clamp_(0, 1)
            img = (img * 255).byte().permute(1, 2, 0).numpy()  # [H,W,3]
            pil = Image.fromarray(img, mode="RGB").resize((Wt, Ht), Image.BILINEAR)
            frames.append(pil)
        return frames

    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        for ch in ["。","！","!","？","?","。","，",",","、"]:
            s = s.replace(ch, "")
        s = " ".join(s.split())
        return s

    def _to_label(self, s: str) -> str:
        # 朴素 token 匹配（已归一化）
        yes = getattr(self, "yes_patterns", {"yes","y","true","1","是","对"})
        no  = getattr(self, "no_patterns",  {"no","n","false","0","否","不对"})
        if any(tok == s or tok in s for tok in self.yes_patterns):
            return "yes"
        if any(tok == s or tok in s for tok in self.no_patterns):
            return "no"
        return "other"

    def _normalize_expected(self, ans: Any) -> List[str]:
        if isinstance(ans, (list, tuple)):
            lst = list(ans)
        else:
            lst = [ans]
        return [self._norm(str(x)) for x in lst if x is not None]