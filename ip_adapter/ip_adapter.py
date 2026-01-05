import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    AutoImageProcessor,
    AutoModel,
)
from safetensors import safe_open
from PIL import Image

from .utils import is_torch2_available, get_generator
from .resampler import Resampler

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
        CNAttnProcessor2_0 as CNAttnProcessor,
        IPAttnProcessor2_0 as IPAttnProcessor,
        IP_CS_AttnProcessor2_0 as IP_CS_AttnProcessor,
        IP_FuAd_AttnProcessor2_0_exp as IP_FuAd_AttnProcessor_exp,
        AttnProcessor2_0_hijack as AttnProcessor_hijack,
        IPAttnProcessor2_0_cross_modal as IPAttnProcessor_cross_modal,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor


DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, signature, logo, lowres, low quality, worst quality, jpeg artifacts, "
    "compression artifacts, blurry, pixelated, noisy, distorted, deformed, disfigured, "
    "bad anatomy, extra limbs, missing parts, broken geometry, washed out, faded colors, "
    "muted colors, oversaturated, undersaturated, color shift, wrong hue, banding, "
    "muddy lighting, dull lighting, harsh shadows, overexposed, underexposed, lens flare, "
    "motion blur, glitch, duplicate objects, messy background"
)


class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, x):
        x = self.proj(x).view(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(x)


class MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            nn.GELU(),
            nn.Linear(clip_embeddings_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, x):
        return self.net(x)


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, target_blocks=("block",)):
        self.device = device
        self.pipe = sd_pipe.to(device)
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks

        self._set_ip_adapter()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path
        ).to(device, dtype=torch.float16)

        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = self._init_proj()
        self._load_ip_adapter()

    def _init_proj(self):
        return ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

    def _set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}

        for name in unet.attn_processors:
            cross_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                hidden = list(reversed(unet.config.block_out_channels))[int(name.split(".")[1])]
            else:
                hidden = unet.config.block_out_channels[int(name.split(".")[1])]

            if cross_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                selected = any(b in name for b in self.target_blocks)
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden,
                    cross_attention_dim=cross_dim,
                    num_tokens=self.num_tokens,
                    scale=1.0,
                    skip=not selected,
                ).to(self.device, dtype=torch.float16)

        unet.set_attn_processor(attn_procs)

        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for net in self.pipe.controlnet.nets:
                    net.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def _load_ip_adapter(self):
        if self.ip_ckpt.endswith(".safetensors"):
            state = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("image_proj."):
                        state["image_proj"][k.replace("image_proj.", "")] = f.get_tensor(k)
                    elif k.startswith("ip_adapter."):
                        state["ip_adapter"][k.replace("ip_adapter.", "")] = f.get_tensor(k)
        else:
            state = torch.load(self.ip_ckpt, map_location="cpu")

        self.image_proj_model.load_state_dict(state["image_proj"])
        nn.ModuleList(self.pipe.unet.attn_processors.values()).load_state_dict(
            state["ip_adapter"], strict=False
        )

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            pixels = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(
                pixels.to(self.device, dtype=torch.float16)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        cond = self.image_proj_model(clip_image_embeds)
        uncond = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return cond, uncond

    def set_scale(self, scale):
        for p in self.pipe.unet.attn_processors.values():
            if isinstance(p, IPAttnProcessor):
                p.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if prompt is None:
            prompt = "best quality"
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        if not isinstance(prompt, list):
            prompt = [prompt]

        cond, uncond = self.get_image_embeds(pil_image, clip_image_embeds)

        cond = cond.repeat_interleave(num_samples, dim=0)
        uncond = uncond.repeat_interleave(num_samples, dim=0)

        with torch.inference_mode():
            pe, ne = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            pe = torch.cat([pe, cond], dim=1)
            ne = torch.cat([ne, uncond], dim=1)

        images = self.pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=get_generator(seed, self.device),
            **kwargs,
        ).images

        return images
