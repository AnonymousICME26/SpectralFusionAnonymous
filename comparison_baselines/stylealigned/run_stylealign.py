import argparse
import os
from typing import Optional

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline
from diffusers.utils import load_image

import inversion
import sa_handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StyleAligned SDXL stylization with reference image."
    )
    parser.add_argument(
        "--style_image",
        type=str,
        required=True,
        help="Path to the reference style image used for inversion.",
    )
    parser.add_argument(
        "--style_prompt",
        type=str,
        required=True,
        help=(
            "Text describing the reference style image (include subject + style). "
            "This is used during DDIM inversion."
        ),
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        required=True,
        help="Prompt describing the target content to generate.",
    )
    parser.add_argument(
        "--style_suffix",
        type=str,
        default=None,
        help=(
            "Optional style phrase appended to the target prompt "
            "(e.g. 'oil painting, impasto brush strokes'). "
            "If omitted, only the provided target_prompt is used."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save the stylized output image.",
    )
    parser.add_argument(
        "--reference_output_path",
        type=str,
        default=None,
        help="Optional path to save the reconstructed reference image (first output).",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL base model identifier or local path.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Optional custom VAE identifier or local path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM steps for inversion and synthesis.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.0,
        help="Classifier-free guidance scale for both inversion and generation.",
    )
    parser.add_argument(
        "--inversion_offset",
        type=int,
        default=5,
        help="Latent offset used when reusing inversion trajectory.",
    )
    parser.add_argument(
        "--shared_score_scale",
        type=float,
        default=1.0,
        help="StyleAligned shared_score_scale parameter.",
    )
    parser.add_argument(
        "--shared_score_shift",
        type=float,
        default=np.log(2.0),
        help="StyleAligned shared_score_shift parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for latent initialization (apart from the reference latent).",
    )
    parser.add_argument(
        "--style_image_size",
        type=int,
        default=1024,
        help="Resolution to which the style image is resized before inversion.",
    )
    return parser.parse_args()


def prepare_pipeline(
    base_model_path: str,
    vae_path: Optional[str],
    device: torch.device,
) -> StableDiffusionXLPipeline:
    is_cuda = device.type == "cuda"
    torch_dtype = torch.float16 if is_cuda else torch.float32

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if is_cuda else None,
        scheduler=scheduler,
    )

    if vae_path:
        vae = AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=torch_dtype, use_safetensors=True
        )
        pipeline.vae = vae

    pipeline = pipeline.to(device)

    if is_cuda:
        pipeline.enable_vae_slicing()

    return pipeline


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device)
    pipeline = prepare_pipeline(args.base_model_path, args.vae_path, device)

    # Load and preprocess the style image.
    style_image = (
        load_image(args.style_image)
        .convert("RGB")
        .resize((args.style_image_size, args.style_image_size))
    )
    style_array = np.array(style_image)

    # Perform DDIM inversion on the style reference.
    inversion_trajectory = inversion.ddim_inversion(
        pipeline,
        style_array,
        args.style_prompt,
        args.num_inference_steps,
        args.guidance_scale,
    )

    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=True,
        share_layer_norm=True,
        share_attention=True,
        adain_queries=True,
        adain_keys=True,
        adain_values=False,
        shared_score_scale=args.shared_score_scale,
        shared_score_shift=args.shared_score_shift,
    )
    handler.register(sa_args)

    zT, inversion_callback = inversion.make_inversion_callback(
        inversion_trajectory, offset=args.inversion_offset
    )

    prompts = [args.style_prompt]
    if args.style_suffix:
        target_prompt = f"{args.target_prompt}, {args.style_suffix}"
    else:
        target_prompt = args.target_prompt
    prompts.append(target_prompt)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

    latent_channels = pipeline.unet.config.in_channels
    latent_size = pipeline.unet.config.sample_size
    latents = torch.randn(
        len(prompts),
        latent_channels,
        latent_size,
        latent_size,
        generator=generator,
        dtype=pipeline.unet.dtype,
    )
    latents = latents.to(device)
    latents[0] = zT.to(device=device, dtype=latents.dtype)

    outputs = pipeline(
        prompts,
        latents=latents,
        callback_on_step_end=inversion_callback,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    ).images

    handler.remove()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    outputs[1].save(args.output_path)

    if args.reference_output_path:
        os.makedirs(os.path.dirname(args.reference_output_path), exist_ok=True)
        outputs[0].save(args.reference_output_path)


if __name__ == "__main__":
    main()

