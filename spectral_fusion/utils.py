import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

from .config import (
    MAX_IMAGE_FILE_SIZE_MB,
    MAX_IMAGE_DIMENSION,
    EPSILON,
)

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def validate_file_size(
    file_path: str,
    max_size_mb: int = MAX_IMAGE_FILE_SIZE_MB,
) -> bool:
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    return file_size_mb <= max_size_mb


def validate_image_dimensions(image: np.ndarray) -> bool:
    h, w = image.shape[:2]
    return h <= MAX_IMAGE_DIMENSION and w <= MAX_IMAGE_DIMENSION


def validate_numeric_parameter(
    value: float,
    name: str,
    min_val: float,
    max_val: float,
) -> float:
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def load_image_safe(
    image_path: str,
    check_size: bool = True,
) -> Optional[np.ndarray]:
    try:
        path = Path(image_path)
        if not path.exists():
            return None

        if check_size and not validate_file_size(str(path)):
            return None

        img = cv2.imread(str(path))
        if img is None:
            return None

        if not validate_image_dimensions(img):
            return None

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception:
        return None


def resize_image_hq(
    image: np.ndarray,
    target_size: int,
    interpolation=cv2.INTER_LANCZOS4,
    keep_aspect: bool = True,
) -> np.ndarray:
    h, w = image.shape[:2]

    if not keep_aspect:
        if h == target_size and w == target_size:
            return image
        return cv2.resize(
            image,
            (target_size, target_size),
            interpolation=interpolation,
        )

    if h == target_size and w == target_size:
        return image

    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image, (new_w, new_h), interpolation=interpolation
    )

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REFLECT_101,
    )

    if padded.shape[0] != target_size or padded.shape[1] != target_size:
        padded = cv2.resize(
            padded,
            (target_size, target_size),
            interpolation=interpolation,
        )

    return padded


def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask

    if gaussian_filter is not None:
        smoothed = gaussian_filter(mask, sigma=sigma)
    else:
        k = max(3, int(round(sigma)))
        if k % 2 == 0:
            k += 1
        smoothed = cv2.GaussianBlur(
            mask.astype(np.float32), (k, k), 0
        )

    return np.clip(smoothed, 0.0, 1.0)


def load_and_preprocess_image(
    image_path: str,
    target_size: int,
) -> Optional[Image.Image]:
    img_array = load_image_safe(image_path)
    if img_array is None:
        return None

    resized = resize_image_hq(img_array, target_size)
    return Image.fromarray(resized)


def validate_image_path(
    image_path: Optional[str],
    arg_name: str = "image",
) -> Optional[str]:
    if image_path is None:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    if not path.is_file():
        return None

    if load_image_safe(str(path)) is None:
        return None

    return str(path)
