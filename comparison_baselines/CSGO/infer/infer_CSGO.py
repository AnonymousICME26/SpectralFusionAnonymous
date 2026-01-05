import argparse
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)

from ip_adapter.utils import BLOCKS, controlnet_BLOCKS, resize_content
from ip_adapter import CSGO


def load_image(path: str, convert_rgb: bool = True) -> Image.Image:
    image = Image.open(path)
    if convert_rgb:
        image = image.convert("RGB")
    return image


def auto_caption(image: Image.Image, device: torch.device) -> str:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    with torch.no_grad():
        inputs = processor(image, return_tensors="pt").to(device)
        generated = model.generate(**inputs)
        caption = processor.decode(generated[0], skip_special_tokens=True)
    return caption


def build_pipe(args: argparse.Namespace, device: torch.device) -> StableDiffusionXLControlNetPipeline:
    vae = None
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)

    pipe_kwargs = dict(
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    if vae is not None:
        pipe_kwargs["vae"] = vae

    if args.controlnet_path:
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_path, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe_kwargs["controlnet"] = controlnet

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model_path,
        **pipe_kwargs,
    )
    pipe.enable_vae_tiling()
    return pipe


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = build_pipe(args, device)

    target_content_blocks = BLOCKS["content"]
    target_style_blocks = BLOCKS["style"]
    controlnet_target_content_blocks = controlnet_BLOCKS["content"]
    controlnet_target_style_blocks = controlnet_BLOCKS["style"]

    csgo = CSGO(
        pipe,
        args.image_encoder_path,
        args.csgo_ckpt,
        device,
        num_content_tokens=args.num_content_tokens,
        num_style_tokens=args.num_style_tokens,
        target_content_blocks=target_content_blocks,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=args.controlnet_path is not None,
        controlnet_target_content_blocks=controlnet_target_content_blocks,
        controlnet_target_style_blocks=controlnet_target_style_blocks,
        content_model_resampler=True,
        style_model_resampler=True,
    )

    style_image = load_image(args.style_path, convert_rgb=True)
    if not args.content_path:
        raise ValueError("CSGO requires a content image. Please provide --content_path.")
    content_image = load_image(args.content_path, convert_rgb=True)

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = auto_caption(content_image, device)
        print(f"BLIP caption: {prompt}")

    num_samples = args.num_samples
    resized_width, resized_height, resized_content = resize_content(content_image)
    print(f"Resized content image to {resized_width}x{resized_height}")

    images = csgo.generate(
        pil_content_image=resized_content,
        pil_style_image=style_image,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=resized_height,
        width=resized_width,
        content_scale=args.content_scale,
        style_scale=args.style_scale,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=num_samples,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        image=resized_content.convert("RGB"),
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    images[0].save(args.output_path)
    print(f"✅ Saved result to {args.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CSGO stylization inference.")

    parser.add_argument("--content_path", type=str, default="./comparison_baselines/CSGO/assets/img_0.png")
    parser.add_argument("--style_path", type=str, default="./comparison_baselines/CSGO/assets/img_1.png")
    parser.add_argument("--output_path", type=str, default="./comparison_baselines/CSGO/assets/output.png")

    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt. If omitted, BLIP captioning is used.")
    parser.add_argument("--negative_prompt", type=str,
                        default="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry")

    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--image_encoder_path", type=str,
                        default="./ip_adapter_models/sdxl_models/image_encoder")
    parser.add_argument("--csgo_ckpt", type=str, default="./InstantX/CSGO/csgo_4_32.bin")
    parser.add_argument("--vae_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--controlnet_path", type=str, default="./ip_adapter_models/controlnet_tile")

    parser.add_argument("--style_scale", type=float, default=1.0)
    parser.add_argument("--content_scale", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.6)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_content_tokens", type=int, default=4)
    parser.add_argument("--num_style_tokens", type=int, default=32)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_inference(args)