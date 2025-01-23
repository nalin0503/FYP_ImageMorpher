import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from accelerate import Accelerator

from model_utility import *
from diffusers.optimization import get_scheduler

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin, AttnProcsLayers
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAAttnAddedKVProcessor,
)
# If you want to use a Karras scheduler for training (rather than DDPMScheduler),
# you can adapt that here. e.g. from diffusers import KarrasDiffusionSchedulers

def train_lora(
    image,
    prompt,
    save_lora_dir,
    model_path=None,
    tokenizer=None,
    text_encoder=None,
    vae=None,
    unet=None,
    noise_scheduler=None,
    lora_steps=200,
    lora_lr=2e-4,
    lora_rank=16,
    weight_name=None,
    safe_serialization=False,
    progress=tqdm
):
    """
    Train a LoRA on a single image (or small dataset) for a given prompt.
    This matches the DiffMorpher approach: freeze most of UNet, VAE, text_encoder,
    then learn rank-limited (LoRA) parameters for cross-attention layers.

    Args:
        image (PIL.Image or np.ndarray):
            The style image you want to capture. If ndarray, it will be converted to PIL.
        prompt (str):
            The textual prompt describing the image style or content.
        save_lora_dir (str):
            Directory where trained LoRA weights will be saved.
        model_path (str, optional):
            Path or HF repo ID containing the base model (if you haven't passed a loaded unet, etc.).
        tokenizer, text_encoder, vae, unet, noise_scheduler:
            Already-initialized components. If None, they will be loaded from model_path.
        lora_steps (int):
            Number of training steps for LoRA.
        lora_lr (float):
            Learning rate for the LoRA parameters.
        lora_rank (int):
            Rank used in LoRA layers.
        weight_name (str, optional):
            Filename for the saved LoRA (e.g. "my_lora.ckpt" or ".safetensors").
        safe_serialization (bool):
            If True, will use safetensors to save the LoRA weights.
        progress:
            TQDM or similar progress bar utility.

    Returns:
        None. Saves a LoRA file to save_lora_dir/weight_name.
    """
    # 1) Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        # optionally: mixed_precision='fp16' or 'bf16'
    )
    # skipping set_seed for the time being #TODO
    # set_seed(0)

    # 2) Load / check tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

    # 3) Load / check scheduler
    if noise_scheduler is None:
        # If you want Karras for training, do (note that karras is just an enum )
        # from diffusers.schedulers import KarrasDiffusionSchedulers
        # noise_scheduler = KarrasDiffusionSchedulers.from_pretrained(model_path, subfolder="scheduler")
        # from diffusers import DPMSolverMultistepScheduler
        # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path,subfolder="scheduler")
        from diffusers import DDIMScheduler
        noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # 4) Load / check text encoder
    if text_encoder is None:
        text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(model_path, subfolder="text_encoder", revision=None)

    # 5) Load / check VAE
    if vae is None:
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=None)

    # 6) Load / check UNet
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=None)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # 7) Initialize UNet LoRA
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        # Cross-attention dimension depends on whether it's self-attn or cross-attn
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        # figure out hidden_size
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks."):].split(".")[0])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks."):].split(".")[0])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("LoRA only set up for up_blocks, mid_block, down_blocks")

        # Distinguish between normal vs. added-KV attention
        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            # fallback to standard LoRAAttnProcessor
            if hasattr(F, "scaled_dot_product_attention"):
                lora_attn_processor_class = LoRAAttnProcessor2_0
            else:
                lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        )

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # 8) Optimizer
    params_to_optimize = unet_lora_layers.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # 9) Learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_steps,
        num_cycles=1,
        power=1.0,
    )

    # 10) Prepare with accelerator
    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    # 11) Text embeddings for the training prompt
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )

    # 12) Preprocess the input image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_transforms = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    image = image_transforms(image).to(device)
    image = image.unsqueeze(dim=0)  # [1,3,512,512]

    latents_dist = vae.encode(image).latent_dist
    # scale factor for SD-based VAE
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    # 13) Actual LoRA training loop
    for step in progress(range(lora_steps), desc="Training LoRA"):
        unet.train()

        # random latents 
        # its adding random noise. 
        model_input = latents_dist.sample() * scaling_factor
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()

        # forward diffusion
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # At minimum, define time_ids so the model doesn't crash:
        time_ids = torch.zeros((bsz, 1), device=text_embedding.device, dtype=torch.int64)

        # If you want time_embeds:
        time_dim = 128
        time_embeds = torch.zeros((bsz, time_dim), device=text_embedding.device, dtype=text_embedding.dtype)
        time_embeds = time_embeds.unsqueeze(1)
        time_embeds = time_embeds.expand(-1, text_embedding.shape[1], -1)

        # Provide the text + time in added_cond_kwargs
        model_pred = unet(
            sample=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=text_embedding,
            added_cond_kwargs={
                "text_embeds": text_embedding,
                "time_embeds": time_embeds,
                "time_ids": time_ids
                
            },
        ).sample

        # MSE loss
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(...)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 14) Save the trained LoRA
    os.makedirs(save_lora_dir, exist_ok=True)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
        weight_name=weight_name,
        safe_serialization=safe_serialization,
    )
    accelerator.end_training()

    print(f"LoRA saved to: {os.path.join(save_lora_dir, weight_name or 'lora.ckpt')}")