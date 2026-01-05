import cv2
import numpy as np
from PIL import Image
from typing import Optional

from .config import (
    SEMANTIC_STYLE_FOREGROUND,
    SEMANTIC_STYLE_BACKGROUND,
    ATTENTION_BLUR_KERNEL,
)
from .utils import resize_image_hq


def compute_attention_map(image: Image.Image) -> np.ndarray:
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    attention = cv2.normalize(
        gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX
    )
    attention = cv2.GaussianBlur(
        attention,
        (ATTENTION_BLUR_KERNEL, ATTENTION_BLUR_KERNEL),
        0,
    )

    return attention


def semantic_aware_fusion(
    generated_image: Image.Image,
    style_reference: Image.Image,
    attention_map: Optional[np.ndarray] = None,
    foreground_style_strength: float = SEMANTIC_STYLE_FOREGROUND,
    background_style_strength: float = SEMANTIC_STYLE_BACKGROUND,
) -> Image.Image:
    img_array = np.array(generated_image).astype(np.float32)
    style_array = np.array(style_reference).astype(np.float32)

    if style_array.shape[:2] != img_array.shape[:2]:
        style_array = resize_image_hq(
            style_array.astype(np.uint8), img_array.shape[1]
        ).astype(np.float32)

    if attention_map is None:
        attention_map = compute_attention_map(generated_image)

    if attention_map.shape[:2] != img_array.shape[:2]:
        attention_map = cv2.resize(
            attention_map,
            (img_array.shape[1], img_array.shape[0]),
        )

    attention_3ch = np.stack([attention_map] * 3, axis=-1)
    background_weight = 1.0 - attention_3ch

    foreground_contribution = img_array * (
        1.0 - foreground_style_strength * attention_3ch
    )
    foreground_style = (
        style_array * foreground_style_strength * attention_3ch
    )

    background_contribution = img_array * (
        1.0 - background_style_strength * background_weight
    )
    background_style = (
        style_array * background_style_strength * background_weight
    )

    result = (
        (foreground_contribution + foreground_style) * attention_3ch
        + (background_contribution + background_style) * background_weight
    )

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)
