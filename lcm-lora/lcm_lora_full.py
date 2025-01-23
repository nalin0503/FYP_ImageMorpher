# lcm_lora_full.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Our LoRA training function
from lora_utility import train_lora

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTokenizer

######################################
# Utility: Slerp & Linear Interp
######################################
def slerp(v0, v1, alpha):
    """
    Spherical linear interpolation for 1D or multi-D PyTorch tensors.
    Fallbacks to linear interpolation if vectors are nearly parallel.
    """
    v0_np = v0.detach().cpu().numpy().astype(np.float32)
    v1_np = v1.detach().cpu().numpy().astype(np.float32)

    dot = (v0_np * v1_np).sum() / (np.linalg.norm(v0_np) * np.linalg.norm(v1_np))
    DOT_THRESHOLD = 0.9995

    if abs(dot) > DOT_THRESHOLD:
        # fallback to linear interpolation
        result = (1 - alpha) * v0_np + alpha * v1_np
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = alpha * theta_0
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = np.sin(theta_t) / sin_theta_0
        result = s0 * v0_np + s1 * v1_np

    return torch.from_numpy(result).to(v0.device, dtype=v0.dtype)


def linear_interp(v0, v1, alpha):
    return (1.0 - alpha) * v0 + alpha * v1

######################################
# Custom LCM-based Inversion Pipeline
######################################
class LCMInversionPipeline(DiffusionPipeline):
    """
    1. Loads a base SDXL model
    2. Uses a Karras-based scheduler (LCM-extended or otherwise)
    3. Provides a method to invert an image to latents (DDIM-like Inversion)
    4. Provides a forward pass with negative_prompt for classifier-free guidance (LCM).
    """

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler  # Karras or LCM-based scheduler

    @torch.no_grad()
    def image2latent(self, image):
        """
        Convert image [-1..1] into latents with factor 0.18215 for SDXL.
        If input is 0..255, first map it to [-1..1].
        """
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image).unsqueeze(0)  # [1,3,H,W]
            image = 2.0 * image - 1.0  # scale to [-1..1]

        image = image.to(self.device, dtype=torch.float32)
        latent_dist = self.vae.encode(image).latent_dist
        latents = latent_dist.mean * 0.18215
        return latents.half()

    @torch.no_grad()
    def latent2image(self, latents):
        """
        Convert latents back to [0..255] images (uint8).
        """
        latents = latents / 0.18215
        image = self.vae.decode(latents.float()).sample  # in [-1..1]
        image = (image / 2 + 0.5).clamp(0, 1).mul(255).type(torch.uint8)
        image = image.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(i) for i in image]

    def inv_step(self, model_output, t, x):
        """
        Inverse sampling step for DDIM-like Inversion.
        model_output: predicted noise at time t
        x: current noisy latents
        """
        # step size: for karras-based scheduler:
        step_size = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        t_next = max(t - step_size, 0)

        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_next = self.scheduler.alphas_cumprod[t_next]
        beta_prod_t = 1 - alpha_prod_t

        # pred_x0 from x_t, noise
        pred_x0 = (x - (beta_prod_t**0.5)*model_output) / (alpha_prod_t**0.5)
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def invert(self, image, prompt, num_inference_steps=50):
        """
        Invert a real image into latents, with no negative prompt usage (guidance_scale=1.0).
        """
        self.scheduler.set_timesteps(num_inference_steps)
        # 1) Convert image to latents
        latents = self.image2latent(image)

        # 2) Encode prompt as normal embeddings (no CF guidance => guidance=1.0)
        text_input = self.tokenizer(
            [prompt] if isinstance(prompt, str) else prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
            truncation=True,
        )
        text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # 3) Walk backward through timesteps to get the "starting noise"
        timesteps = list(reversed(self.scheduler.timesteps))
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion")):
            # single pass => no uncond/cond chunking
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeds).sample
            latents, _ = self.inv_step(noise_pred, t, latents)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        latents=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
    ):
        """
        Forward pass (LCM-based) with negative_prompt for classifier-free guidance.
        If latents is None, we do random latents in [B,unet_in_channels,H//8,W//8].
        """
        self.scheduler.set_timesteps(num_inference_steps)

        # Prepare latents if user didn't supply them
        if latents is None:
            latents = torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
                dtype=torch.float16
            )
        else:
            if latents.ndim == 3:
                latents = latents.unsqueeze(0)
            latents = latents.to(self.device, dtype=torch.float16)

        # Prepare text embeddings for CFG if guidance_scale>1
        if prompt is None:
            prompt = ""
        text_input = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
        text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]

        if guidance_scale > 1.0:
            if negative_prompt:
                neg_input = self.tokenizer([negative_prompt], padding="max_length", max_length=77, return_tensors="pt")
                neg_embeds = self.text_encoder(neg_input.input_ids.to(self.device))[0]
            else:
                # unconditional
                neg_input = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
                neg_embeds = self.text_encoder(neg_input.input_ids.to(self.device))[0]
            # cat
            text_embeds = torch.cat([neg_embeds, text_embeds], dim=0)

        # Forward sampling
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps, desc="Forward (LCM) Sampling")):
            if guidance_scale > 1.0:
                model_in = torch.cat([latents]*2, dim=0)
            else:
                model_in = latents

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                noise_pred = self.unet(model_in, t, encoder_hidden_states=text_embeds).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)

            # do step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode
        images = self.latent2image(latents)
        return images


############################################################
# Putting it all together: LCM + LoRAs + interpolation
############################################################
def main():
    """
    1) Optionally train 2 LoRAs (if not found).
    2) Load base SDXL pipeline with Karras/LCM scheduler.
    3) Create custom LCMInversionPipeline for inversion + forward pass.
    4) Invert two real images w/ guidance_scale=1.0 (no negative prompt).
    5) SLERP latents and do LCM-based forward pass with negative prompt if desired.
    6) Save frames and optionally animate them.
    """
    device = "cuda"

    # Base model ID
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    # 1) Prepare LoRA training or loading
    style_lora_dir = "my_lora_dir"
    os.makedirs(style_lora_dir, exist_ok=True)

    # Suppose we have two style images
    styleA_img = "../assets/vangogh.jpg"
    styleB_img = "../assets/pearlgirl.jpg"

    styleA_prompt = "oil painting of a man"
    styleB_prompt = "oil painting of a woman"

    styleA_ckpt = os.path.join(style_lora_dir, "styleA.safetensors")
    styleB_ckpt = os.path.join(style_lora_dir, "styleB.safetensors")

    # If not present, train them
    if not os.path.exists(styleA_ckpt):
        imgA = Image.open(styleA_img).convert("RGB")
        train_lora(imgA, styleA_prompt, style_lora_dir, model_path=base_id, weight_name="styleA.safetensors")

    if not os.path.exists(styleB_ckpt):
        imgB = Image.open(styleB_img).convert("RGB")
        train_lora(imgB, styleB_prompt, style_lora_dir, model_path=base_id, weight_name="styleB.safetensors")

    # 2) Load base pipeline w/ Karras + LCM for forward
    pipe = DiffusionPipeline.from_pretrained(
        base_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    # pipe.scheduler = KarrasDiffusionSchedulers.from_config(pipe.scheduler.config)
    # If you have an official LCM scheduler:
    from diffusers import LCMScheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load LoRAs (including an LCM "acceleration vector" if needed)
    # Example:
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    pipe.load_lora_weights(styleA_ckpt, adapter_name="styleA")
    pipe.load_lora_weights(styleB_ckpt, adapter_name="styleB")

    # 3) Create custom pipeline
    custom_pipe = LCMInversionPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    custom_pipe.to(device)

    # 4) Invert two real images
    imageA_path = "../assets/vangogh.jpg"
    imageB_path = "../assets/pearlgirl.jpg"
    if not os.path.exists(imageA_path) or not os.path.exists(imageB_path):
        print("Warning: 'imageA.jpg' or 'imageB.jpg' not found, using placeholders.")
    imageA = Image.open(imageA_path).convert("RGB").resize((512, 512))
    imageB = Image.open(imageB_path).convert("RGB").resize((512, 512))

    # invert with guidance_scale=1 => no negative prompt
    latentsA = custom_pipe.invert(imageA, prompt="A photo of image A", num_inference_steps=50)
    latentsB = custom_pipe.invert(imageB, prompt="A photo of image B", num_inference_steps=50)

    # 5) Interpolate latents
    def latent_interp(a, b, alpha, use_slerp=True):
        if use_slerp:
            return slerp(a, b, alpha)
        else:
            return linear_interp(a, b, alpha)

    # 6) For each alpha, set LoRA blend, do forward pass
    frames = []
    out_dir = "morph_results"
    os.makedirs(out_dir, exist_ok=True)

    num_frames = 10
    for i in range(num_frames):
        alpha = i / (num_frames - 1) if num_frames > 1 else 0.0
        # blend styleA vs styleB
        wA = 1.0 - alpha
        wB = alpha
        # If using an "acceleration vector," do: 
        pipe.set_adapters(["lcm","styleA","styleB"], [1.0, wA, wB])
        # pipe.set_adapters(["styleA","styleB"], [wA, wB])

        latentsT = latent_interp(latentsA, latentsB, alpha, use_slerp=True)

        # do forward pass with negative prompt if desired
        negative_prompt = "blurry, low quality, oversaturated, random, incoherent"  # undesired features
        images = custom_pipe(
            prompt="A final, consistent morph of these two styles",
            negative_prompt=negative_prompt,
            latents=latentsT,
            num_inference_steps=20,
            guidance_scale=5.0,
            height=512,
            width=512,
        )
        frame = images[0]
        frame.save(os.path.join(out_dir, f"frame_{i:03d}.png"))
        frames.append(frame)

    # Optional: build GIF
    if len(frames) > 1:
        frames[0].save(
            os.path.join(out_dir, "morph.gif"),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
        print(f"Saved morph GIF at {os.path.join(out_dir, 'morph.gif')}")

    print("Done morphing! Check the frames in:", out_dir)


if __name__ == "__main__":
    main()
