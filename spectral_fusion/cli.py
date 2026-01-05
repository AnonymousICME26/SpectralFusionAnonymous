import argparse
import sys
from .config import (
    DEFAULT_STYLE_SIZE,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STYLE_SCALE,
    DEFAULT_POST_PROCESS_STRENGTH,
    FDSM_STRUCTURE_WEIGHT,
    FDSM_TEXTURE_WEIGHT,
    PLR_NUM_STAGES,
)


def create_parser():
    parser = argparse.ArgumentParser(
        description="SpectralFusion: Frequency-Domain Visual Stylization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--style_path", type=str, required=True)

    parser.add_argument("--neg_style_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./output.jpg")

    parser.add_argument("--style_size", type=int, default=DEFAULT_STYLE_SIZE)
    parser.add_argument("--output_size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--style_scale", type=float, default=DEFAULT_STYLE_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default="")

    enhancement_group = parser.add_argument_group("Expert Enhancements")
    enhancement_group.add_argument(
        "--quality_mode",
        type=str,
        choices=["fast", "balanced", "high", "ultra", "perfect", "research"],
        default=None,
    )
    enhancement_group.add_argument("--auto_optimize", action="store_true")
    enhancement_group.add_argument("--enhance_style", action="store_true")
    enhancement_group.add_argument("--multiscale_style", action="store_true")
    enhancement_group.add_argument("--adaptive_schedule", action="store_true")
    enhancement_group.add_argument(
        "--schedule_type",
        type=str,
        default="progressive",
        choices=["progressive", "cosine", "inverse"],
    )
    enhancement_group.add_argument("--post_process", action="store_true")
    enhancement_group.add_argument(
        "--post_process_strength",
        type=float,
        default=DEFAULT_POST_PROCESS_STRENGTH,
    )

    week2_group = parser.add_argument_group("Week 2 Enhancements")
    week2_group.add_argument("--semantic_aware", action="store_true")
    week2_group.add_argument("--semantic_fg_strength", type=float, default=0.9)
    week2_group.add_argument("--semantic_bg_strength", type=float, default=1.0)
    week2_group.add_argument("--advanced_color", action="store_true")
    week2_group.add_argument(
        "--color_method",
        type=str,
        default="reinhard",
        choices=["reinhard", "linear", "mkl"],
    )
    week2_group.add_argument("--color_strength", type=float, default=0.4)
    week2_group.add_argument("--style_mixing", action="store_true")
    week2_group.add_argument("--style_paths", type=str, nargs="+", default=None)
    week2_group.add_argument("--mixing_weights", type=float, nargs="+", default=None)
    week2_group.add_argument(
        "--mixing_strategy",
        type=str,
        default="weighted",
        choices=["weighted", "hierarchical", "adaptive"],
    )

    novel_group = parser.add_argument_group("SpectralFusion Techniques")
    novel_group.add_argument("--fdsm", action="store_true")
    novel_group.add_argument(
        "--fdsm_structure", type=float, default=FDSM_STRUCTURE_WEIGHT
    )
    novel_group.add_argument(
        "--fdsm_texture", type=float, default=FDSM_TEXTURE_WEIGHT
    )
    novel_group.add_argument("--saf", action="store_true")
    novel_group.add_argument("--plr", action="store_true")
    novel_group.add_argument("--plr_stages", type=int, default=PLR_NUM_STAGES)
    novel_group.add_argument("--hcp", action="store_true")
    novel_group.add_argument("--hcp_strength", type=float, default=0.8)
    novel_group.add_argument("--adaptive_layers", action="store_true")

    stylestudio_group = parser.add_argument_group("StyleStudio Comparison Mode")
    stylestudio_group.add_argument("--stylestudio_mode", action="store_true")
    stylestudio_group.add_argument("--fuAttn", action="store_true")
    stylestudio_group.add_argument("--fuSAttn", action="store_true")
    stylestudio_group.add_argument("--fuIPAttn", action="store_true")
    stylestudio_group.add_argument("--fuScale", type=float, default=1.0)
    stylestudio_group.add_argument("--end_fusion", type=int, default=0)
    stylestudio_group.add_argument("--adainIP", action="store_true")

    advanced_group = parser.add_argument_group("Advanced Quality Improvements")
    advanced_group.add_argument("--noise_scale", type=float, default=0.95)
    advanced_group.add_argument("--noise_offset", type=float, default=0.02)
    advanced_group.add_argument("--dynamic_balance", action="store_true")
    advanced_group.add_argument(
        "--content_aware_fdsm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    advanced_group.add_argument("--perceptual_boost", action="store_true")

    model_group = parser.add_argument_group("Model Paths")
    model_group.add_argument("--base_model_path", type=str, default=None)
    model_group.add_argument("--vae_path", type=str, default=None)
    model_group.add_argument("--image_encoder_path", type=str, default=None)
    model_group.add_argument("--csgo_ckpt", type=str, default=None)

    return parser


def apply_quality_preset(args):
    if args.quality_mode is None:
        return args

    presets = {
        "fast": {
            "num_inference_steps": 30,
            "style_size": 512,
            "style_scale": 0.9,
            "enhance_style": False,
            "auto_optimize": False,
            "post_process": False,
            "multiscale_style": False,
            "adaptive_schedule": False,
            "semantic_aware": False,
            "advanced_color": False,
        },
        "balanced": {
            "num_inference_steps": 50,
            "style_size": 768,
            "style_scale": 1.0,
            "guidance_scale": 7.5,
            "enhance_style": True,
            "post_process": True,
            "post_process_strength": 0.2,
            "multiscale_style": True,
        },
        "high": {
            "num_inference_steps": 70,
            "style_size": 768,
            "style_scale": 1.2,
            "guidance_scale": 7.5,
            "enhance_style": True,
            "post_process": True,
            "post_process_strength": 0.25,
            "multiscale_style": True,
        },
        "ultra": {
            "num_inference_steps": 100,
            "style_size": 768,
            "style_scale": 1.5,
            "guidance_scale": 8.0,
            "enhance_style": True,
            "post_process": True,
            "post_process_strength": 0.3,
            "multiscale_style": True,
            "schedule_type": "progressive",
        },
        "perfect": {
            "num_inference_steps": 90,
            "style_size": 768,
            "style_scale": 1.3,
            "guidance_scale": 8.0,
            "enhance_style": True,
            "post_process": True,
            "post_process_strength": 0.25,
            "multiscale_style": True,
        },
        "research": {
            "num_inference_steps": 100,
            "style_size": 1024,
            "style_scale": 0.6,
            "guidance_scale": 10.5,
            "auto_optimize": True,
            "post_process": True,
            "post_process_strength": 0.25,
            "multiscale_style": True,
            "adaptive_schedule": True,
            "semantic_aware": True,
            "fdsm_structure": 1.6,
            "fdsm_texture": 0.45,
            "saf": True,
            "hcp": True,
            "hcp_strength": 0.75,
            "noise_scale": 1.0,
            "noise_offset": 0.02,
            "dynamic_balance": True,
            "content_aware_fdsm": True,
        },
    }

    preset = presets[args.quality_mode]

    for key, value in preset.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args


def parse_args():
    parser = create_parser()
    args = parser.parse_args()

    cli_args = sys.argv[1:]
    args._style_scale_cli = "--style_scale" in cli_args
    args._guidance_scale_cli = "--guidance_scale" in cli_args
    args._num_steps_cli = "--num_inference_steps" in cli_args

    args = apply_quality_preset(args)
    return args
