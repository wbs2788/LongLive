# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import datasets
from typing import List, Dict, Optional, Any
from collections import defaultdict
import random

class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }

class TwoTextQADataset(Dataset):
    """
    Dataset that returns two text prompts + optional QA list per sample.

    - Keeps the behavior of TwoTextDataset (two prompts per sample).
    - Adds QA reading from a JSONL file whose path comes from config.
      Each line looks like:
        {"video_id": "vid_0000", "type": "quality",
         "question": "...?", "answer": "Yes"}

    Mapping from sample idx -> video_id is provided by:
      - a line-aligned file (video_id_list_path), or
      - a format string video_id_format (e.g. "vid_{:04d}").

    Args:
        prompt_path (str): first-segment prompts, one per line.
        switch_prompt_path (str): second-segment prompts, one per line.
        qa_jsonl_path (str): path from config for QA jsonl.
        video_id_list_path (Optional[str]): optional path, one video_id per line, aligned with prompts.
        video_id_format (str): fallback formatter if no video_id_list provided, default "vid_{:04d}".
        qa_type_filter (Optional[List[str]]): keep only these types if provided.
        qa_per_sample (Optional[int]): if set, sample at most this many QAs per sample.
        shuffle_qa (bool): whether to shuffle QA list before sub-sampling.
    """
    def __init__(
        self,
        prompt_path: str,
        switch_prompt_path: str,
        qa_jsonl_path: Optional[str] = None,
        video_id_list_path: Optional[str] = None,
        video_id_format: str = "vid_{:04d}",
        qa_type_filter: Optional[List[str]] = None,
        qa_per_sample: Optional[int] = None,
        shuffle_qa: bool = True,
    ):
        # ==== prompts ====
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip("\n") for line in f]
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip("\n") for line in f]
        assert len(self.switch_prompt_list) == len(self.prompt_list), \
            "prompt_path & switch_prompt_path must have same number of lines."

        self.N = len(self.prompt_list)

        # ==== idx -> video_id ====
        self.video_ids: Optional[List[str]] = None
        self.video_id_format = video_id_format
        if video_id_list_path is not None:
            with open(video_id_list_path, encoding="utf-8") as f:
                self.video_ids = [line.strip() for line in f]
            assert len(self.video_ids) == self.N, \
                "video_id_list must align 1-1 with prompts."

        # ==== QA jsonl ====
        self.qa_by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.qa_per_sample = qa_per_sample
        self.shuffle_qa = shuffle_qa
        self.qa_type_filter = set(qa_type_filter) if qa_type_filter else None

        if qa_jsonl_path is not None:
            with open(qa_jsonl_path, encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Bad JSON at line {ln} in {qa_jsonl_path}: {e}") from e

                    vid = obj.get("video_id", None)
                    qtype = obj.get("type", None)
                    question = obj.get("question", None)
                    answer = obj.get("answer", None)

                    if vid is None or question is None or answer is None:
                        raise ValueError(f"Missing required fields at line {ln} in {qa_jsonl_path}")

                    if self.qa_type_filter and (qtype not in self.qa_type_filter):
                        continue

                    self.qa_by_vid[vid].append({
                        "type": qtype,
                        "question": question,
                        "answer": answer
                    })


    def __len__(self):
        return self.N

    def _idx_to_video_id(self, idx: int) -> str:
        if self.video_ids is not None:
            return self.video_ids[idx]
        # fallback：用格式字符串从 idx 生成，例如 "vid_0000"
        return self.video_id_format.format(idx)

    def __getitem__(self, idx):
        vid = self._idx_to_video_id(idx)
        qa_list = self.qa_by_vid.get(vid, [])

        # 子采样（可选）
        if self.qa_per_sample is not None and len(qa_list) > self.qa_per_sample:
            if self.shuffle_qa:
                # 随机挑选 qa_per_sample 条
                qa_list = random.sample(qa_list, k=self.qa_per_sample)
            else:
                # 固定顺序的前 k 条（可确保可复现）
                qa_list = qa_list[: self.qa_per_sample]

        return {
            "prompts": self.prompt_list[idx],                 # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],   # second-segment prompt
            "video_id": vid,
            "qa": qa_list,  # List[{"type": str, "question": str, "answer": str}]
            "idx": idx,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        简单的 collate_fn：把列表对齐为 batch 字典。qa 是变长 list，保持原样。
        你也可以在这里做 QA 的 tokenization/编码。
        """
        out = {
            "prompts": [b["prompts"] for b in batch],
            "switch_prompts": [b["switch_prompts"] for b in batch],
            "video_id": [b["video_id"] for b in batch],
            "idx": [b["idx"] for b in batch],
            "qa": [b["qa"] for b in batch],  # List[List[Dict]]
        }
        return out
class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
