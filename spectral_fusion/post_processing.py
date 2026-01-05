import cv2
import numpy as np
from PIL import Image, ImageFilter

from .config import (
    DEFAULT_POST_PROCESS_STRENGTH,
    TEXTURE_BLEND_WEIGHT,
    COLOR_BLEND_WEIGHT,
    EPSILON,
)
from .utils import validate_numeric_parameter, resize_image_hq


def post_process_stylized_image(
    image: Image.Image,
    style_reference: Image.Image,
    strength: float = DEFAULT_POST_PROCESS_STRENGTH,
) -> Image.Image:
    strength = validate_numeric_parameter(
        strength, "post_process_strength", 0.0, 1.0
    )

    img_array = np.array(image)
    style_array = np.array(style_reference)

    if style_array.shape[:2] != img_array.shape[:2]:
        style_array = resize_image_hq(style_array, img_array.shape[1])

    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    style_lab = cv2.cvtColor(style_array, cv2.COLOR_RGB2LAB).astype(np.float32)

    for i in range(3):
        img_mean = img_lab[:, :, i].mean()
        img_std = img_lab[:, :, i].std()
        style_mean = style_lab[:, :, i].mean()
        style_std = style_lab[:, :, i].std()

        img_lab[:, :, i] = (
            ((img_lab[:, :, i] - img_mean) / (img_std + EPSILON))
            * (style_std + EPSILON)
            * strength
            + style_mean * strength
            + img_lab[:, :, i] * (1.0 - strength)
        )

    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    color_matched = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(color_matched, cv2.COLOR_RGB2GRAY)
    high_freq = cv2.Laplacian(gray, cv2.CV_64F)
    high_freq = cv2.normalize(
        high_freq, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    enhanced = color_matched.copy()
    for i in range(3):
        enhanced[:, :, i] = cv2.addWeighted(
            color_matched[:, :, i],
            COLOR_BLEND_WEIGHT,
            high_freq,
            TEXTURE_BLEND_WEIGHT * 1.2,
            0,
        )

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    pil_enhanced = Image.fromarray(enhanced)
    final = pil_enhanced.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2)
    )

    return final


def selective_sharpen(
    image: Image.Image,
    amount: float = 0.35,
    radius: float = 2.0,
    softness_threshold: float = 180.0,
) -> Image.Image:
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var >= softness_threshold:
        return image

    blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
    sharpened = cv2.addWeighted(
        img_array, 1.0 + amount, blurred, -amount, 0
    )
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)
