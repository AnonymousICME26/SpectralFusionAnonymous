import cv2
import numpy as np
from PIL import Image
from typing import Optional

from .config import EPSILON
from .utils import resize_image_hq


def advanced_color_grading(
    image: Image.Image,
    style_reference: Image.Image,
    method: str = "reinhard",
    strength: float = 0.6,
) -> Image.Image:
    img_array = np.array(image).astype(np.float32) / 255.0
    style_array = np.array(style_reference).astype(np.float32) / 255.0

    if style_array.shape[:2] != img_array.shape[:2]:
        style_array = cv2.resize(
            style_array, (img_array.shape[1], img_array.shape[0])
        )

    if method == "reinhard":
        img_lab = cv2.cvtColor(
            (img_array * 255).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        style_lab = cv2.cvtColor(
            (style_array * 255).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)

        result_lab = img_lab.copy()
        for i in range(3):
            img_mean = img_lab[:, :, i].mean()
            img_std = img_lab[:, :, i].std()
            style_mean = style_lab[:, :, i].mean()
            style_std = style_lab[:, :, i].std()

            result_lab[:, :, i] = (
                (img_lab[:, :, i] - img_mean)
                * (style_std + EPSILON)
                / (img_std + EPSILON)
                + style_mean
            )

        result_lab = np.clip(result_lab, 0, 255)
        result = (
            cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            .astype(np.float32)
            / 255.0
        )

    elif method == "linear":
        result = np.zeros_like(img_array)
        for i in range(3):
            img_channel = img_array[:, :, i].flatten()
            style_channel = style_array[:, :, i].flatten()

            matched = np.interp(
                img_channel,
                np.percentile(img_channel, np.linspace(0, 100, 256)),
                np.percentile(style_channel, np.linspace(0, 100, 256)),
            )
            result[:, :, i] = matched.reshape(img_array.shape[:2])

    elif method == "mkl":
        result = img_array.copy()
        scales = [1, 3, 5]
        for scale in scales:
            img_blur = cv2.GaussianBlur(img_array, (0, 0), scale)
            style_blur = cv2.GaussianBlur(style_array, (0, 0), scale)

            for i in range(3):
                img_mean = img_blur[:, :, i].mean()
                style_mean = style_blur[:, :, i].mean()
                result[:, :, i] += (style_mean - img_mean) / len(scales)

        result = np.clip(result, 0, 1)

    else:
        result = img_array

    final = img_array * (1.0 - strength) + result * strength
    final = np.clip(final * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(final)


def mix_multiple_styles(
    style_images: list,
    weights: Optional[list] = None,
    mixing_strategy: str = "weighted",
    target_size: int = 1024,
) -> Image.Image:
    from .config import MAX_STYLE_REFERENCES
    from .style_analysis import analyze_style_intensity

    if len(style_images) < 2:
        return style_images[0]

    if len(style_images) > MAX_STYLE_REFERENCES:
        style_images = style_images[:MAX_STYLE_REFERENCES]

    if weights is None:
        weights = [1.0 / len(style_images)] * len(style_images)
    else:
        total = sum(weights[: len(style_images)])
        if abs(total) < 1e-8:
            weights = [1.0 / len(style_images)] * len(style_images)
        else:
            weights = [w / total for w in weights[: len(style_images)]]

    if mixing_strategy == "weighted":
        mixed_array = None
        for style_img, weight in zip(style_images, weights):
            style_array = np.array(style_img).astype(np.float32)
            if style_array.shape[0] != target_size:
                style_array = resize_image_hq(
                    style_array.astype(np.uint8), target_size
                ).astype(np.float32)

            if mixed_array is None:
                mixed_array = style_array * weight
            else:
                mixed_array += style_array * weight

        return Image.fromarray(
            np.clip(mixed_array, 0, 255).astype(np.uint8)
        )

    elif mixing_strategy == "hierarchical":
        base_array = np.array(style_images[0]).astype(np.float32)
        if base_array.shape[0] != target_size:
            base_array = resize_image_hq(
                base_array.astype(np.uint8), target_size
            ).astype(np.float32)

        base_low = cv2.GaussianBlur(base_array, (0, 0), 5)

        for style_img, weight in zip(style_images[1:], weights[1:]):
            style_array = np.array(style_img).astype(np.float32)
            if style_array.shape[0] != target_size:
                style_array = resize_image_hq(
                    style_array.astype(np.uint8), target_size
                ).astype(np.float32)

            style_low = cv2.GaussianBlur(style_array, (0, 0), 5)
            style_high = style_array - style_low
            base_low += style_high * weight

        return Image.fromarray(
            np.clip(base_low, 0, 255).astype(np.uint8)
        )

    elif mixing_strategy == "adaptive":
        style_weights = []
        for style_img in style_images:
            analysis = analyze_style_intensity(style_img)
            style_weights.append(analysis["intensity"])

        total_weight = sum(style_weights)
        style_weights = [w / total_weight for w in style_weights]

        if weights:
            combined = [sw * uw for sw, uw in zip(style_weights, weights)]
            total = sum(combined)
            combined = [w / total for w in combined]
        else:
            combined = style_weights

        mixed_array = None
        for style_img, weight in zip(style_images, combined):
            style_array = np.array(style_img).astype(np.float32)
            if style_array.shape[0] != target_size:
                style_array = resize_image_hq(
                    style_array.astype(np.uint8), target_size
                ).astype(np.float32)

            if mixed_array is None:
                mixed_array = style_array * weight
            else:
                mixed_array += style_array * weight

        return Image.fromarray(
            np.clip(mixed_array, 0, 255).astype(np.uint8)
        )

    else:
        return mix_multiple_styles(
            style_images, weights, "weighted", target_size
        )
