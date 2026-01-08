import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk, load_dataset # Added for HF datasets

from pipeline import (
    CausalInferencePipeline,
)
# from utils.dataset import TextDataset # No longer needed
from utils.misc import set_seed

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

# ----------------- New Dataset Wrapper -----------------
class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        print(f"Loading dataset from: {dataset_path}")
        try:
            # Try loading as a saved arrow dataset (load_from_disk)
            self.data = load_from_disk(dataset_path)["test"]
        except Exception:
            try:
                # Fallback: try loading as a directory/script
                self.data = load_dataset(dataset_path)["test"]
            except Exception as e:
                print(f"Error loading dataset: {e}")
                raise e

        # Handle DatasetDict (if 'train' split exists)
        if hasattr(self.data, 'keys') and not hasattr(self.data, 'features'):
            if 'train' in self.data.keys():
                self.data = self.data['train']
            else:
                # Fallback to the first available split
                key = list(self.data.keys())[0]
                self.data = self.data[key]
        
        # Automatically determine the text column name
        self.text_col = 'text'
        if len(self.data) > 0:
            sample = self.data[0]
            # Common names for text-to-video datasets
            candidates = ['prompt', 'caption', 'text', 'prompts', 'long_prompt']
            for c in candidates:
                if c in sample:
                    self.text_col = c
                    break
        
        print(f"Dataset loaded. Size: {len(self.data)}. Using column '{self.text_col}' for prompts.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = item[self.text_col]
        
        # Ensure it's a string
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        # Return dict matching original TextDataset format
        return {
            'idx': idx,
            'prompts': prompt_text,
            'extended_prompts': prompt_text # Use same text if no extension logic exists
        }
# -------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)


# Initialize pipeline
# Note: checkpoint loading is now handled inside the pipeline __init__ method
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # Apply LoRA to generator transformer
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights (if provided)
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True


# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------- Modified: Load Local HuggingFace Dataset -----------------
dataset_path = "../wbs/moviegen"
dataset = HFDatasetWrapper(dataset_path)
num_prompts = len(dataset)
# ----------------------------------------------------------------------------

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

# ----------------- Modified: TQDM Progress Bar -----------------
# Added `total=len(dataloader)` and `desc` for better visibility
for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating Videos", disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch
    
    prompt = batch['prompts'][0]

    check_base_name = prompt.replace('/', '').replace('\\', '')
    if len(check_base_name) > 40:
        check_base_name = check_base_name[:40]
    
    # Check if all seeds for this prompt exist
    all_files_exist = True
    for seed_idx in range(config.num_samples):
        check_filename = f"{check_base_name}-{seed_idx}.mp4"
        check_filepath = os.path.join(config.output_folder, check_filename)
        if not os.path.exists(check_filepath):
            all_files_exist = False
            break
    
    if all_files_exist:
        if local_rank == 0:
            # print(f"Skipping existing: {check_base_name}...") # Optional: Comment out to reduce spam in progress bar
            pass
        continue
    
    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    # For text-to-video, batch is just the text prompt
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None and len(extended_prompt) > 0:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    # Clean up logs to keep progress bar clean
    # print("sampled_noise.device", sampled_noise.device)
    # print("prompts", prompts)

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        base_prompt_for_name = prompt 

        # Sanitize filename
        base_prompt_for_name = base_prompt_for_name.replace('/', '').replace('\\', '')
        for bad in ('/', '\\'):
            assert bad not in base_prompt_for_name, f"prompt contains illegal chars {bad}"
        if len(base_prompt_for_name) > 40:
            base_prompt_for_name = base_prompt_for_name[:40]
            
        for seed_idx in range(config.num_samples):
            out_name = f"{base_prompt_for_name}-{seed_idx}.mp4" 
            output_path = os.path.join(config.output_folder, out_name)
            write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
        
if dist.is_initialized():
    dist.destroy_process_group()