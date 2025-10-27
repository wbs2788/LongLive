# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Adopted from Self-Forcing / LongLive
#
# Purpose
# -------
# Extend DMD with a GRPO-weighted training path using an external
# Video-QA Judge (VQJ). The judge turns your Yes/No QA into a pass-rate
# reward per candidate video. We compute group-relative advantages, map
# them to positive weights, and multiply those weights onto the standard
# DMD loss. No change to DMD math; only a wrapper around rollout/loss.
#
# Naming
# ------
# VQJ = Video-QA Judge (replaces "VLA").
#
# Requirements
# ------------
# - DMD._run_generator(...) exists and returns:
#     pred_image [B, F, C, H, W] in latent space,
#     gradient_mask (or None),
#     denoised_timestep_from, denoised_timestep_to
# - A VAE or decoder available as self.vae.decode_to_pixel(latents, use_cache=False)
#   (or adjust `_decode_to_rgb` to your stack).
#
# How to use
# ----------
# 1) Instantiate DMDGRPOVQJ(dmd_args, device, vqj=YourJudge()).
# 2) Call grpo_weighted_dmd_step(...) inside your training loop.
#    It returns (loss, log_dict). Backprop on loss as usual.

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from model.vqj import VideoQAJudge
from model.dmd import DMD


# =========================
# GRPO configuration & helpers
# =========================

@dataclass
class GRPOConfig:
    group_size: int = 4        # number of candidates per prompt
    beta: float = 6.0          # temperature for sigmoid weighting
    alpha: float = 0.5         # minimum weight (avoid zero gradient)
    clip_c: float = 0.25       # clamp advantage to [-c, +c]
    use_ema_baseline: bool = False
    ema_momentum: float = 0.9  # EMA for baseline across steps

def _map_adv_to_weight(
    adv: torch.Tensor, beta: float, alpha: float, clip_c: float
) -> torch.Tensor:
    """
    Map advantages to positive sample weights:
    w = alpha + sigmoid(beta * clip(adv, -c, +c))
    """
    adv = torch.clamp(adv, -clip_c, clip_c)
    return alpha + torch.sigmoid(beta * adv)


# =========================
# DMD + GRPO + VQJ
# =========================

class DMDGRPOVQJ(DMD):
    """
    DMD extension with grouped generation and VQJ-based GRPO reweighting.
    This class does NOT change DMD math; it only orchestrates:
      (1) Generate G candidates via _run_generator
      (2) Decode to RGB and score with VQJ (pass-rate rewards)
      (3) Compute group-relative advantages and weights
      (4) Reuse compute_distribution_matching_loss per candidate,
          multiply by the sample weight, and sum.

    Use this class when you want to align generation with your QA signals.
    """

    def __init__(
        self,
        args,
        device,
        vqj: Optional[VideoQAJudge] = None,
        grpo_cfg: Optional[GRPOConfig] = None,
    ):
        super().__init__(args, device)
        self.vqj = vqj
        self.grpo_cfg = grpo_cfg or GRPOConfig()
        self._ema_baseline: Dict[str, float] = {}

    # -------- decoding helper (adjust to your stack) --------
    @torch.no_grad()
    def _decode_to_rgb(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert latents [B, F, C, H, W] to pixel videos in [0,1], shape [B, F, 3, H, W].
        Adapt this to your VAE / decoder. If your generator already outputs pixels,
        just return the input or the proper transform.
        """
        # Try to use self.vae if available (LongLive-style)
        if hasattr(self, "vae") and hasattr(self.vae, "decode_to_pixel"):
            vid = self.vae.decode_to_pixel(latents, use_cache=False)
            vid = (vid * 0.5 + 0.5).clamp(0, 1)
            return vid
        # Fallback: assume latents are already pixel space in [-1,1]
        vid = (latents * 0.5 + 0.5).clamp(0, 1)
        return vid

    # -------- baseline & advantages --------
    def _group_baseline(self, rewards: torch.Tensor, key: Optional[str]) -> torch.Tensor:
        if not self.grpo_cfg.use_ema_baseline or key is None:
            return rewards.mean()
        old = self._ema_baseline.get(key, float(rewards.mean().item()))
        new = self.grpo_cfg.ema_momentum * old + (1 - self.grpo_cfg.ema_momentum) * float(rewards.mean().item())
        self._ema_baseline[key] = new
        return torch.tensor(new, device=rewards.device, dtype=rewards.dtype)

    def _advantages_and_weights(
        self, rewards: torch.Tensor, prompt_key: Optional[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        baseline = self._group_baseline(rewards, prompt_key)
        adv = rewards - baseline
        w = _map_adv_to_weight(adv, self.grpo_cfg.beta, self.grpo_cfg.alpha, self.grpo_cfg.clip_c)
        return adv, w

    # -------- main: grouped rollout + judge + weighted DMD loss --------
    def grpo_weighted_dmd_step(
        self,
        prompt_key: str,
        # generator inputs
        image_or_video_shape,                 # [B(=G), F, C, H, W] shape spec for _run_generator
        conditional_dict: dict,
        unconditional_dict: dict,
        # judge inputs
        qa_list: List[Dict[str, Any]],
        # options
        initial_latents: Optional[List[torch.Tensor]] = None,  # optional seeds for each candidate
        return_videos: bool = False,                            # for logging/inspection
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate G candidates (group), compute pass-rate rewards with VQJ,
        form GRPO weights, then sum weighted DMD losses.

        Returns:
          total_loss: scalar tensor
          log_dict:   misc scalars and per-candidate info (for logging)
        """
        assert self.vqj is not None, "Please provide a VideoQAJudge (vqj) implementation."
        G = self.grpo_cfg.group_size
        device = self.device

        # Containers
        cand_latents = []     # list of [1, F, C, H, W]
        masks = []            # per-candidate gradient_mask (or None)
        denoise_from = []     # ints
        denoise_to = []       # ints

        # --- (1) Roll out G candidates via DMD's internal generator path ---
        # We reuse your existing backward-simulated generator to obtain latents.
        for i in range(G):
            # If you want different seeds per candidate, pass initial_latent[i]
            init_latent_i = None if initial_latents is None else initial_latents[i]

            pred_img_i, mask_i, t_from_i, t_to_i = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=init_latent_i,
                slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            )
            # Ensure batch dim = 1 per candidate for simplicity
            if pred_img_i.shape[0] != 1:
                # If your _run_generator returns B>1, split
                pred_img_i = pred_img_i[:1]
                if mask_i is not None: mask_i = mask_i[:1]
            cand_latents.append(pred_img_i)   # [1, F, C, H, W]
            masks.append(mask_i)
            denoise_from.append(t_from_i)
            denoise_to.append(t_to_i)

        # --- (2) Decode to RGB and get VQJ rewards ---
        with torch.no_grad():
            # Stack to [G, F, C, H, W] and decode
            latents_stack = torch.cat(cand_latents, dim=0)                    # [G, F, C, H, W]
            videos_rgb = self._decode_to_rgb(latents_stack).detach()          # [G, F, 3, H, W], [0,1]

            rewards = []
            judge_items = []
            for i in range(G):
                # Judge expects [T, 3, H, W]
                score_out = self.vqj.score(videos_rgb[i], qa_list)
                rewards.append(float(score_out.get("pass_rate", 0.0)))
                judge_items.append(score_out)

            rewards_t = torch.tensor(rewards, device=device, dtype=latents_stack.dtype)  # [G]
            adv_t, w_t = self._advantages_and_weights(rewards_t, prompt_key)

        # --- (3) Compute per-candidate DMD loss and weight it ---
        total_loss = 0.0
        per_cand_losses = []
        for i in range(G):
            # DMD’s KL/score-matching loss on predicted latents (teacher vs. student)
            dmd_loss_i, _ = self.compute_distribution_matching_loss(
                image_or_video=cand_latents[i],                   # [1, F, C, H, W]
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                gradient_mask=masks[i],
                denoised_timestep_from=denoise_from[i],
                denoised_timestep_to=denoise_to[i]
            )
            # Multiply by stop-grad GRPO weight
            total_loss = total_loss + (w_t[i].detach() * dmd_loss_i)
            per_cand_losses.append(float(dmd_loss_i.detach().cpu()))

        log = {
            "grpo/avg_reward": float(rewards_t.mean().cpu()),
            "grpo/avg_weight": float(w_t.mean().cpu()),
            "grpo/adv_mean": float(adv_t.mean().cpu()),
            "grpo/adv_std": float(adv_t.std().cpu()),
            "grpo/per_cand_loss": per_cand_losses,
            "grpo/per_cand_reward": rewards,
        }
        if return_videos:
            # Warning: logging videos can be heavy; keep for debugging only.
            log["grpo/sample_videos_rgb"] = videos_rgb[:min(2, G)].detach().cpu()  # small preview

        return total_loss, log
