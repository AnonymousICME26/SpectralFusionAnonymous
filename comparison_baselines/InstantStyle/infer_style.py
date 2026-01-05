import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterXL


def load_image(image_path: Path) -> Image.Image:
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((512, 512))
    return pil_image


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_path = args.base_model_path or "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = args.image_encoder_path or "./ip_adapter_models/sdxl_models/image_encoder"
    ip_ckpt = args.ip_ckpt or "./ip_adapter_models/sdxl_models/ip-adapter_sdxl.bin"
    vae_path = args.vae_path

    vae = None
    if vae_path:
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

    pipe_kwargs = dict(
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    if vae is not None:
        pipe_kwargs["vae"] = vae

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        **pipe_kwargs,
    )
    pipe.enable_vae_tiling()

    ip_model = IPAdapterXL(
        pipe,
        image_encoder_path,
        ip_ckpt,
        device,
        target_blocks=args.target_blocks,
    )

    style_image = load_image(Path(args.style_path))

    images = ip_model.generate(
        pil_image=style_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        scale=args.style_scale,
        guidance_scale=args.guidance_scale,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        neg_content_prompt=args.neg_content_prompt,
        neg_content_scale=args.neg_content_scale,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"✅ Saved result to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InstantStyle SDXL inference.")

    parser.add_argument("--style_path", type=str, default="./comparison_baselines/InstantStyle/assets/0.jpg")
    parser.add_argument("--prompt", type=str, default="a cat, masterpiece, best quality, high quality")
    parser.add_argument("--negative_prompt", type=str,
                        default="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry")
    parser.add_argument("--output_path", type=str, default="./comparison_baselines/InstantStyle/result.png")

    parser.add_argument("--style_scale", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--neg_content_prompt", type=str, default=None)
    parser.add_argument("--neg_content_scale", type=float, default=0.5)

    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Local path or repo id for SDXL base model.")
    parser.add_argument("--image_encoder_path", type=str, default=None,
                        help="Path to CLIP image encoder for IP-Adapter.")
    parser.add_argument("--ip_ckpt", type=str, default=None,
                        help="Path to IP-Adapter checkpoint (.bin or .safetensors).")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Optional SDXL VAE path.")
    parser.add_argument("--target_blocks", nargs="+", default=["up_blocks.0.attentions.1"],
                        help="UNet attention blocks to inject style tokens.")

    args = parser.parse_args()
    main(args)