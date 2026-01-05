import os
import math
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

from ip_adapter.utils import BLOCKS
from ip_adapter import SpectralFusion_Engine
from ip_adapter.adaptive_layer_fusion import create_adaptive_layer_fusion

from spectral_fusion.utils import (
    validate_image_path,
    validate_numeric_parameter,
    load_and_preprocess_image,
    resize_image_hq,
)
from spectral_fusion.style_analysis import (
    analyze_style_intensity,
    enhance_style_image,
    load_multiscale_style,
    blend_multiscale_features,
    AdaptiveStyleController,
)
from spectral_fusion.post_processing import (
    post_process_stylized_image,
    selective_sharpen,
)
from spectral_fusion.semantic import semantic_aware_fusion
from spectral_fusion.color import (
    advanced_color_grading,
    mix_multiple_styles,
)
from spectral_fusion.techniques import (
    apply_fdsm,
    apply_saf,
    apply_plr,
    apply_hcp,
)
from spectral_fusion.config import (
    device,
    weight_dtype,
    DEFAULT_STYLE_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    MIN_GUIDANCE_SCALE,
    DEFAULT_LATENT_CHANNELS,
    VAE_DOWNSCALE_FACTOR,
    AUTO_OPT_STYLE_DELTA,
    AUTO_OPT_GUIDANCE_DELTA,
    AUTO_OPT_STEPS_DELTA,
    MAX_STYLE_REFERENCES,
    MAX_SIMULTANEOUS_TECHNIQUES,
    EARLY_STYLE_BOOST,
    LATE_STYLE_REDUCTION,
    DEFAULT_CONTRAST_FACTOR,
    MULTISCALE_WEIGHTS,
    FDSM_STRUCTURE_WEIGHT,
    FDSM_TEXTURE_WEIGHT,
    PLR_NUM_STAGES,
)

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    import pywt
except ImportError:
    pywt = None


def main(args):
    style_path = validate_image_path(args.style_path)
    neg_style_path = validate_image_path(args.neg_style_path)

    args.style_scale = validate_numeric_parameter(args.style_scale, "", 0.1, 3.0)
    args.guidance_scale = validate_numeric_parameter(
        args.guidance_scale, "", MIN_GUIDANCE_SCALE, 20.0
    )
    args.post_process_strength = validate_numeric_parameter(
        args.post_process_strength, "", 0.0, 1.0
    )
    args.num_inference_steps = int(
        validate_numeric_parameter(args.num_inference_steps, "", 10, 200)
    )

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image_encoder_path = (
        getattr(args, "image_encoder_path", None)
        or "./ip_adapter_models/sdxl_models/image_encoder"
    )
    csgo_ckpt = getattr(args, "csgo_ckpt", None) or "InstantX/CSGO/csgo_4_32.bin"
    base_model_path = (
        getattr(args, "base_model_path", None)
        or "stabilityai/stable-diffusion-xl-base-1.0"
    )
    vae_path = (
        getattr(args, "vae_path", None) or "madebyollin/sdxl-vae-fp16-fix"
    )

    try:
        vae = AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=weight_dtype
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=weight_dtype,
            add_watermarker=False,
            vae=vae,
        ).to(device)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )

        pipe.enable_vae_tiling()
        if device.type == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
    except Exception:
        sys.exit(1)

    try:
        spectral_engine = SpectralFusion_Engine(
            pipe,
            image_encoder_path,
            csgo_ckpt,
            device,
            num_style_tokens=32,
            target_style_blocks=BLOCKS["style"],
            controlnet_adapter=False,
            style_model_resampler=True,
            fuAttn=getattr(args, "fuAttn", False),
            fuSAttn=getattr(args, "fuSAttn", False),
            fuIPAttn=getattr(args, "fuIPAttn", False),
            fuScale=args.style_scale,
            end_fusion=getattr(args, "end_fusion", 0),
            adainIP=getattr(args, "adainIP", False),
            weight_dtype=weight_dtype,
        )
        adaptive_fusion = create_adaptive_layer_fusion()
    except Exception:
        sys.exit(1)

    generator = torch.Generator(device.type).manual_seed(args.seed)
    latent_size = args.output_size // VAE_DOWNSCALE_FACTOR

    init_latents = torch.randn(
        (1, DEFAULT_LATENT_CHANNELS, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=weight_dtype,
    ) * getattr(args, "noise_scale", 1.0)

    if getattr(args, "noise_offset", 0.0) > 0:
        init_latents = init_latents + args.noise_offset

    if (
        getattr(args, "style_mixing", False)
        and getattr(args, "style_paths", None)
        and len(args.style_paths) >= 2
    ):
        styles = []
        for p in args.style_paths[:MAX_STYLE_REFERENCES]:
            img = load_and_preprocess_image(p, args.style_size)
            if img is not None:
                styles.append(img)
        if len(styles) >= 2:
            style_image = mix_multiple_styles(
                styles,
                weights=getattr(args, "mixing_weights", None),
                mixing_strategy=getattr(args, "mixing_strategy", "weighted"),
                target_size=args.style_size,
            )
        else:
            sys.exit(1)
    else:
        style_image = load_and_preprocess_image(style_path, args.style_size)
        if style_image is None:
            sys.exit(1)

    if getattr(args, "auto_optimize", False):
        analysis = analyze_style_intensity(style_image)
        args._spectral_style_analysis = analysis

        if not getattr(args, "_style_scale_cli", False):
            args.style_scale = float(
                np.clip(
                    analysis["recommended_scale"],
                    max(0.1, DEFAULT_STYLE_SCALE - AUTO_OPT_STYLE_DELTA),
                    DEFAULT_STYLE_SCALE + AUTO_OPT_STYLE_DELTA,
                )
            )

        if not getattr(args, "_guidance_scale_cli", False):
            args.guidance_scale = float(
                np.clip(
                    analysis["recommended_cfg"],
                    max(
                        MIN_GUIDANCE_SCALE,
                        DEFAULT_GUIDANCE_SCALE - AUTO_OPT_GUIDANCE_DELTA,
                    ),
                    DEFAULT_GUIDANCE_SCALE + AUTO_OPT_GUIDANCE_DELTA,
                )
            )

        if not getattr(args, "_num_steps_cli", False):
            args.num_inference_steps = int(
                np.clip(
                    analysis["recommended_steps"],
                    max(10, DEFAULT_NUM_INFERENCE_STEPS - AUTO_OPT_STEPS_DELTA),
                    DEFAULT_NUM_INFERENCE_STEPS + AUTO_OPT_STEPS_DELTA,
                )
            )

    if getattr(args, "enhance_style", False):
        style_image = enhance_style_image(
            style_image, DEFAULT_CONTRAST_FACTOR
        )

    multiscale_enabled = False
    if getattr(args, "multiscale_style", False):
        scales = load_multiscale_style(style_path)
        if len(scales) >= 2:
            style_image = blend_multiscale_features(
                scales, MULTISCALE_WEIGHTS
            )
            if style_image.size[0] != args.style_size:
                style_image = Image.fromarray(
                    resize_image_hq(
                        np.array(style_image), args.style_size
                    )
                )
            multiscale_enabled = True

    neg_style = (
        load_and_preprocess_image(neg_style_path, args.style_size)
        if neg_style_path
        else None
    )

    adaptive_controller = None
    if getattr(args, "adaptive_schedule", False):
        adaptive_controller = AdaptiveStyleController(
            args.style_scale,
            args.num_inference_steps,
            getattr(args, "schedule_type", "progressive"),
        )

    images = spectral_engine.generate(
        pil_style_image=style_image,
        neg_pil_style_image=neg_style,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.output_size,
        width=args.output_size,
        style_scale=args.style_scale,
        guidance_scale=args.guidance_scale,
        guidance_rescale=0.7,
        num_images_per_prompt=1,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        latents=init_latents,
        adaptive_controller=adaptive_controller,
    )

    final_image = images[0]
    reference = final_image.copy()

    if getattr(args, "post_process", False):
        final_image = post_process_stylized_image(
            final_image,
            style_image,
            args.post_process_strength,
        )

    if getattr(args, "semantic_aware", False):
        final_image = semantic_aware_fusion(
            final_image,
            style_image,
            foreground_style_strength=getattr(
                args, "semantic_fg_strength", 0.9
            ),
            background_style_strength=getattr(
                args, "semantic_bg_strength", 1.0
            ),
        )

    if getattr(args, "advanced_color", False):
        final_image = advanced_color_grading(
            final_image,
            style_image,
            method=getattr(args, "color_method", "reinhard"),
            strength=getattr(args, "color_strength", 0.4),
        )

    if getattr(args, "fdsm", False) and gaussian_filter is not None:
        final_image = apply_fdsm(
            final_image,
            style_image,
            getattr(args, "fdsm_structure", FDSM_STRUCTURE_WEIGHT),
            getattr(args, "fdsm_texture", FDSM_TEXTURE_WEIGHT),
            getattr(args, "content_aware_fdsm", True),
        )

    if getattr(args, "saf", False) and multiscale_enabled and pywt is not None:
        scales = load_multiscale_style(style_path)
        if len(scales) >= 2:
            fused = apply_saf(list(scales.values()))
            final_image = apply_fdsm(
                final_image,
                fused,
                FDSM_STRUCTURE_WEIGHT,
                FDSM_TEXTURE_WEIGHT,
                True,
            )

    if getattr(args, "plr", False):
        final_image = apply_plr(
            final_image,
            getattr(args, "plr_stages", PLR_NUM_STAGES),
        )

    if getattr(args, "hcp", False):
        final_image = apply_hcp(
            final_image,
            reference,
            getattr(args, "hcp_strength", 0.8),
        )

    final_image = selective_sharpen(final_image)
    final_image.save(args.output_path, quality=95, optimize=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from spectral_fusion.cli import parse_args

    main(parse_args())
